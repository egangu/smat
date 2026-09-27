"""Scale and Mask state shared by the reference and fused implementations."""

import math
import re
import torch


def embedding_parameter_names(model, named_parameters=None):
    """Include all aliases of an embedding weight, including a tied LM head."""
    ids = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    parameters = (
        model.named_parameters() if named_parameters is None else named_parameters
    )
    return {name for name, parameter in parameters if id(parameter) in ids}


def block_linear_parameter_names(model, named_parameters=None):
    """Attention/MLP matrices only: exclude embeddings, heads, bias and norms."""
    embedding_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)
    }
    ids = {
        id(m.weight)
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and re.match(r".*(?:encoder|model)\.layers\.\d+\.", name)
        and id(m.weight) not in embedding_ids
    }
    parameters = (
        model.named_parameters() if named_parameters is None else named_parameters
    )
    return {name for name, parameter in parameters if id(parameter) in ids}


class ScaleMask:
    """Cache the immutable initialization and independent Scale/Mask streams."""

    def __init__(
        self,
        parameters,
        *,
        alpha_min,
        probability,
        seed,
        block_linear_names,
        compact_base,
    ):
        if not math.isfinite(alpha_min) or not 0 <= alpha_min <= 1:
            raise ValueError("scale.alpha_min must be finite and in [0,1]")
        if not math.isfinite(probability) or not 0 <= probability < 1:
            raise ValueError("mask.probability must be finite and in [0,1)")
        self.minimum = alpha_min
        self.scope = "global"
        self.mask_rescale = True
        self.mask_kind = "bernoulli"
        self.mask_granularity = "coordinate"
        self.noise_coupling = "constant"
        self.block_linear_names = set(block_linear_names)
        if probability and not self.block_linear_names:
            raise ValueError("Mask requires selected Attention/MLP linear weights")
        self.base = {
            name: p.detach().to(
                dtype=p.dtype if compact_base else torch.float32, copy=True
            )
            for name, p in parameters
        }
        self.mask_probabilities = {
            name: probability if name in self.block_linear_names else 0.0
            for name, _ in parameters
        }
        self.mask_enabled = any(self.mask_probabilities.values())
        self.mask_active = False
        self.mask_steps = 0
        self.generator = torch.Generator().manual_seed(seed + 1)
        self.mask_seed = seed + 3
        self.mask_generators = {}
        self.masks = {}

    def sample(self):
        return (
            self.minimum
            + (1 - self.minimum) * torch.rand((), generator=self.generator).item()
        )

    def scale(self, name, global_scale):
        return global_scale

    def begin_mask_step(self, enabled=True):
        self.mask_active = enabled and self.mask_enabled
        self.mask_steps += int(self.mask_active)

    def has_mask(self, name):
        return self.mask_active and self.mask_probabilities[name] > 0

    def sample_mask(self, name, parameter):
        device = str(parameter.device)
        if device not in self.mask_generators:
            self.mask_generators[device] = torch.Generator(
                device=parameter.device
            ).manual_seed(self.mask_seed)
        if name not in self.masks:
            self.masks[name] = torch.empty(
                parameter.shape, device=parameter.device, dtype=torch.bool
            )
        mask = self.masks[name]
        mask.bernoulli_(
            1 - self.mask_probabilities[name], generator=self.mask_generators[device]
        )
        return mask

    def value(self, name, parameter, scale):
        base = self.base[name]
        if self.has_mask(name):
            delta = parameter.float() - base
            mask = self.sample_mask(name, parameter)
            return (
                delta.mul(mask)
                .mul_(scale / (1 - self.mask_probabilities[name]))
                .add_(base)
            )
        if scale == 1.0:
            return parameter.to(dtype=torch.float32, copy=True)
        return torch.lerp(base, parameter.float(), scale)
