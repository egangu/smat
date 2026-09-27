"""Periodic single-pass scheduling (Algorithm 1 in the paper)."""


class SmatSchedule:
    """Apply the selected operators every t-th update; other steps are FT."""

    def __init__(self, specification, *, available, seed):
        spec = dict(specification or {})
        if set(spec) - {"interval", "mode", "components"}:
            raise ValueError("smat accepts interval, mode and components")
        self.interval = spec.get("interval", 4)
        if type(self.interval) is not int or self.interval < 1:
            raise ValueError("smat.interval must be a positive integer")
        self.mode = spec.get("mode", "joint")
        if self.mode != "joint":
            raise ValueError("the released schedule uses joint periodic application")
        self.components = tuple(spec.get("components", available))
        if len(set(self.components)) != len(self.components) or not set(
            self.components
        ) <= set(available):
            raise ValueError(f"components must be unique and available: {available}")
        self.scheduled_steps = self.clean_steps = 0
        self.counts = dict.fromkeys(available, 0)

    def select(self, step_index):
        active = set()
        if (step_index + 1) % self.interval == 0:
            self.scheduled_steps += 1
            active = set(self.components)
        self.clean_steps += int(not active)
        for component in active:
            self.counts[component] += 1
        return active

    def metadata(self):
        return dict(
            interval=self.interval,
            components=list(self.components),
            scheduled_steps=self.scheduled_steps,
            clean_steps=self.clean_steps,
            operator_steps=dict(self.counts),
        )
