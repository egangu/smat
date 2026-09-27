"""Sparse task-vector mergers with explicit, fixed settings.

TIES and DARE follow FusionBench 54c9e8c9d9621620c720452cd8533332a32d3689
(method/ties_merging and method/dare). DELLA follows declare-lab/della's
rank_magnitude sampler and normalized sign-consensus aggregation. DELLA rank
and probability arithmetic uses FP32, including for BF16 source parameters.
"""

from contextlib import ExitStack
import json
import math
from pathlib import Path
import torch


def ties_update(deltas, threshold=20):
    """Global per-expert magnitude threshold, global zero-sign rule, sum merge."""
    keep = threshold / 100 if threshold > 1 else threshold
    if not 0 < keep <= 1:
        raise ValueError("TIES retention must be in (0, 1]")
    k = max(1, deltas.shape[1] - int(deltas.shape[1] * keep))
    cut = deltas.abs().kthvalue(k, dim=1, keepdim=True).values
    trimmed = deltas * (deltas.abs() >= cut)
    signs = torch.sign(trimmed.sum(dim=0))
    signs[signs == 0] = torch.sign(signs.sum())
    selected = torch.where(signs.unsqueeze(0) > 0, trimmed > 0, trimmed < 0)
    return (trimmed * selected).sum(dim=0)


def dare_prune(delta, drop=0.5):
    """Same dtype random samples and operation order as FusionBench DARE."""
    return (delta * (torch.rand_like(delta) > drop)) / (1 - drop)


def della_prune(delta, drop=0.3, window=0.14, *, uniforms=None):
    """Row rank -> retention probability -> Bernoulli -> inverse rescaling.

    Vectors are treated as one row. For higher-rank tensors, the last axis
    defines a row; a length-one row receives the mean retention probability.
    Optional uniforms allow a small exact-sample reference check.
    """
    density = 1 - drop
    if not 0 < density - window / 2 <= density + window / 2 < 1:
        raise ValueError("DELLA retention probability must stay inside (0, 1)")
    values = delta.float().reshape(-1, delta.shape[-1] if delta.ndim else 1)
    width = values.shape[-1]
    if width > 1:
        order = values.abs().argsort(dim=1)
        ranks = torch.empty_like(values)
        ranks.scatter_(
            1,
            order,
            torch.arange(width, device=values.device, dtype=torch.float32).expand_as(
                values
            ),
        )
        probability = density - window / 2 + ranks / (width - 1) * window
    else:
        probability = torch.full_like(values, density)
    mask = (
        torch.bernoulli(probability)
        if uniforms is None
        else (uniforms.reshape_as(values) < probability)
    )
    return (values * mask / probability).reshape_as(delta)


def sparse_merge(
    adapter,
    base,
    experts,
    output,
    *,
    method,
    scale,
    seed=42,
    state_dict_aliases=None,
    ties_retention=0.2,
    dare_drop=0.5,
    della_drop=0.3,
    della_window=0.14,
):
    """Reuse the existing checkpoint scope/assets; no training or evaluation."""
    if method not in ("ties", "dare", "della"):
        raise ValueError(method)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("merge scale must be finite and positive")
    if not 0 < ties_retention <= 1 or not 0 <= dare_drop < 1:
        raise ValueError("invalid TIES retention or DARE drop probability")
    if (
        not 0 <= della_window
        or not 0
        < 1 - della_drop - della_window / 2
        <= 1 - della_drop + della_window / 2
        < 1
    ):
        raise ValueError("DELLA retention probability must stay inside (0, 1)")
    torch.manual_seed(seed)
    with ExitStack() as stack:
        anchor = stack.enter_context(adapter.read(base))
        sources = [stack.enter_context(adapter.read(p)) for p in experts]
        writer = stack.enter_context(adapter.write(output, template=base))
        names = sorted(anchor.keys())
        if any(set(s.keys()) != set(names) for s in sources):
            raise ValueError("Base and experts have different checkpoint scopes")
        if method == "ties":
            # HF safetensors omit tied aliases; FusionBench flattens state_dict,
            # which includes them. Count aliases for threshold/sign statistics,
            # but write only the canonical checkpoint keys.
            aliases = state_dict_aliases
            model_config = (
                Path(base) / "config.json" if isinstance(base, (str, Path)) else None
            )
            if aliases is None and model_config is not None and model_config.is_file():
                tied = json.loads(model_config.read_text()).get(
                    "tie_word_embeddings", False
                )
                if (
                    tied
                    and "lm_head.weight" not in names
                    and "model.embed_tokens.weight" in names
                ):
                    aliases = {"lm_head.weight": "model.embed_tokens.weight"}
            aliases = aliases or {}
            if any(n in names or target not in names for n, target in aliases.items()):
                raise ValueError("Invalid or redundant state_dict alias")
            full_names = sorted([*names, *aliases])
            floating = [
                n
                for n in full_names
                if anchor.get_tensor(aliases.get(n, n)).is_floating_point()
            ]
            base_vector = torch.cat(
                [anchor.get_tensor(aliases.get(n, n)).reshape(-1) for n in floating]
            )
            vectors = torch.stack(
                [
                    torch.cat(
                        [s.get_tensor(aliases.get(n, n)).reshape(-1) for n in floating]
                    )
                    for s in sources
                ]
            )
            vectors.sub_(base_vector)
            update = ties_update(vectors, threshold=ties_retention)
            del vectors
            merged = base_vector + scale * update
            offset = 0
            for name in full_names:
                value = anchor.get_tensor(aliases.get(name, name))
                if value.is_floating_point():
                    size = value.numel()
                    value = merged[offset : offset + size].reshape_as(value)
                    offset += size
                if name not in aliases:
                    writer.put_tensor(name, value)
            return
        for name in names:
            value = anchor.get_tensor(name)
            if not value.is_floating_point():
                writer.put_tensor(name, value)
                continue
            if method == "dare":
                # Streaming task order matches state_dict_sum; source dtype is preserved.
                total = None
                for source in sources:
                    delta = dare_prune(source.get_tensor(name) - value, drop=dare_drop)
                    total = delta.clone() if total is None else total.add_(delta)
                merged = value + scale * total
            else:
                # Process row batches on CPU to bound temporary rank memory.
                original_shape = value.shape
                width = original_shape[-1] if value.ndim else 1
                base_rows = value.reshape(-1, width)
                source_rows = [s.get_tensor(name).reshape(-1, width) for s in sources]
                merged_rows = torch.empty_like(base_rows)
                for start in range(0, len(base_rows), 256):
                    stop = start + 256
                    base_part = base_rows[start:stop].float()
                    deltas = torch.stack(
                        [
                            della_prune(
                                s[start:stop].float() - base_part,
                                drop=della_drop,
                                window=della_window,
                            )
                            for s in source_rows
                        ]
                    )
                    sign = torch.where(deltas.sum(0) >= 0, 1, -1)
                    mask = deltas.sign() == sign
                    mixed = (deltas * mask).sum(0) / mask.sum(0).clamp_min(1)
                    merged_rows[start:stop] = (base_part + scale * mixed).to(
                        value.dtype
                    )
                merged = merged_rows.reshape(original_shape)
            writer.put_tensor(name, merged)
