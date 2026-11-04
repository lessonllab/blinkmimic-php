"""Bound MuJoCo Warp's narrowphase reads in PHP mesh probes."""

import torch
import warp as wp
from mujoco_warp._src import collision_driver


@wp.kernel
def _bound_candidate_count(
    count: wp.array(dtype=wp.int32), capacity: int, peak: wp.array(dtype=wp.int32)
):
    peak[0] = wp.max(peak[0], count[0])
    count[0] = wp.min(count[0], capacity)


_narrowphase = collision_driver._narrowphase


def _guarded_narrowphase(model, data, context):
    peak = getattr(data, "_php_probe_candidate_peak", None)
    if peak is not None:
        # The broadphase counter includes discarded pairs. The convex kernel
        # uses this counter as a loop bound even when the pair array is full.
        wp.launch(
            _bound_candidate_count,
            dim=1,
            inputs=[data.ncollision, context.collision_pair.shape[0], peak],
            device=data.ncollision.device,
        )
    return _narrowphase(model, data, context)


def guard_probe_collision(data):
    """Record the raw candidate peak before clamping the GPU loop bound."""
    peak = torch.zeros(1, dtype=torch.int32, device=str(data.ncollision.device))
    data._php_probe_candidate_peak = wp.from_torch(peak)
    collision_driver._narrowphase = _guarded_narrowphase
    return peak
