"""Optional execution optimizations; network and checkpoint formats stay unchanged."""

import torch


def configure_adam(optimizer, *, fused: bool, capturable=False):
    """Restore execution flags after load_state_dict overwrites parameter groups.

    Older checkpoints store CPU step counters, whereas fused CUDA Adam needs
    them on the parameter device. Moments and step counts are otherwise kept.
    """
    for group in optimizer.param_groups:
        group.update(fused=fused, foreach=None, capturable=capturable)
        if capturable:
            group['lr'] = torch.as_tensor(group['lr'], device=group['params'][0].device).clone()
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "step" in state:
                state["step"] = state["step"].to(
                    device=parameter.device if fused else "cpu"
                )


def finite_adam_step(optimizer, grad_norm, skipped):
    """Gate the entire fused Adam update on-device, including moments and steps.

    Zeroing bad gradients alone would still move parameters through momentum.
    Fused Adam's AMP gate skips all state changes without a CUDA scalar read.
    """
    found_inf = (~torch.isfinite(grad_norm)).float()
    skipped.add_(found_inf.to(skipped.dtype))
    optimizer.found_inf = found_inf
    try:
        optimizer.step()
    finally:
        optimizer.found_inf = None


def sum_metrics(values, device):
    """Reduce a logging window on-device before the single distributed read."""
    values = tuple(values)
    if not values:
        return torch.zeros((), device=device, dtype=torch.float64)
    return torch.stack(
        [torch.as_tensor(v, device=device, dtype=torch.float64) for v in values]
    ).sum()


class GraphedUpdate:
    """Capture a deterministic update with fixed tensor inputs, then replay it.

    Sampling stays outside the graph. Warm-up/capture executions are rolled
    back in-place, including optimizer moments and counters, so the first call
    performs exactly ONE real update. In-place restore preserves graph pointers.
    """

    def __init__(self, fn, module, optimizer, counters=()):
        self.fn, self.module, self.optimizer = fn, module, optimizer
        self.counters = tuple(counters)
        self.graph = None

    def _capture(self, inputs):
        self.inputs = {key: value.detach().clone() for key, value in inputs.items()}
        tensors = list(self.module.state_dict().values()) + list(self.counters)
        tensors += [v for state in self.optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
        tensors += [g['lr'] for g in self.optimizer.param_groups if torch.is_tensor(g['lr'])]
        saved = [(t, t.detach().clone()) for t in tensors]
        stream = torch.cuda.Stream(device=next(self.module.parameters()).device)
        current = torch.cuda.current_stream(stream.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.fn(self.inputs)
        current.wait_stream(stream)
        self._restore(saved)
        stream.wait_stream(current)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = self.fn(self.inputs)
        current.wait_stream(stream)
        self._restore(saved)

    @staticmethod
    @torch.no_grad()
    def _restore(saved):
        for tensor, value in saved:
            tensor.copy_(value)

    def __call__(self, inputs):
        if self.graph is None:
            self._capture(inputs)
        for key, value in inputs.items():
            self.inputs[key].copy_(value)
        self.graph.replay()
        # Each invocation must own its metrics; replay overwrites static outputs.
        return tuple(value.clone() for value in self.outputs)
