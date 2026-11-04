# BlinkMimic PHP adaptation

Source: https://github.com/fuyichen0725/blinkmimic
Revision: `d264658d6c9ce61fb52f9f0f8985a369f3ccc409`.
The upstream Python package is vendored here with MIT/third-party notices;
`scripts/blinkmimic-presets.sh` preserves the released recipes. The launcher
copies `PRESET[main]` exactly before applying explicit PHP/hardware overrides.

## Retained training method

- World model: 1024/1024/1024 MLP, semi-implicit structured integration,
  eight-step fit window, five fits per update group, batch 2048, mirrored state,
  action and simulation context for half of each batch.
- Policy: original 2048/1024/512/256 encoder/decoder, reference-pose residual
  action, no world-model-only warmup, imagined batch 2048 and horizon 32.
- Muon on the upstream-selected hidden weight matrices of both models; Adam
  elsewhere and on the critic. Policy EMA 0.995 is the deployed/evaluated policy.
- Learning rate 4e-4, policy multiplier 4, progress-triggered decay from return
  0.75 (by group 1400) to multiplier 1 over 300 groups; native base schedule.
- Native return-based sampling, duplicate-bin fix, recent-failure reset share
  0.3, probe-guided reset share 0.15, starting at group 1000 every 125 groups,
  failure-count decay 0.7, 10-second pre-failure window.
- Compiled model/MDP functions, fused rotations/Adam, training CUDA graphs,
  collection/masked-reset graph, probe graph, no-sync normalizers, and background
  light checkpoint writes. Original objective/GAE/critic updates are retained.

## PHP task adaptations

`php/adapter.py` bridges the existing named NPZ loader to BlinkMimic's state,
goal, future-frame and bin contracts. It loads all four available PHP skills,
balances the initial skill mass, retains 50 Hz poses/velocities and clip ends,
and uses the 29-DoF sphere-hand G1 assets. Actor observation is 99 dimensions;
ten future reference frames give a 730-dimensional goal. New weights cannot
load old PHP teacher checkpoints.

Seven padded OBB terrain slots are real, independently resized/moved collision
objects in every world, updated on both eager and masked graph resets. Ten
variants per motion remain available during training; evaluation uses variant
0. Raw simulation context has 103 fields (33 randomization fields + 70 terrain
fields). World model and critic encode each active box as heading-relative
position, rotation-6D, size and occupancy. This preserves translation/yaw
invariance; mirror augmentation also mirrors the obstacle positions/rotations.
The actor uses proprioception and reference goals, as in BlinkMimic; this is
not a visual navigation policy.

The base tracking reward and torso/hand/foot failure thresholds retain the PHP
task semantics. BlinkMimic's contact matching/anti-slip imagination reward
uses support height from an OBB ray intersection rather than a flat floor,
and calibrates the reference-contact clearance per clip. Noisy/randomized
training and deterministic evaluation use the upstream sensing paths.

`args.json`/checkpoint metadata record source revision, actual flags, dimensions,
joint/body names and SHA-256 hashes for the motion/terrain pairs. Evaluation
and playback reject mismatched formats/data rather than loading incompatible
weights silently. A group is five world-model updates plus one policy/critic
update and an average 2.5 simulator vector steps (UTD 0.4).

## Explicit local differences

- 2048 simulator worlds and a replay capacity of 512 vector steps to fit the
  local 16 GiB RTX 5080. The logical training batches remain 2048.
- Actor BPTT runs in four chunks of 512, accumulating each chunk's mean
  gradient with weight 512/2048 before a single global clip, Muon/Adam step and
  EMA update. The critic normalizer and GAE/critic schedule still see the full
  2048 batch. This changes floating-point accumulation order and noise draws,
  not the objective or number of updates. A deterministic test executes the
  actual training functions to compare actor, critic, normalizer and EMA state
  against an unchunked update. Set `--pi_microbatch 0` to disable chunking when
  enough memory is available. The unchunked upstream-default BF16 graph exceeded
  this machine's shared-GPU memory during validation. Network MLP matmuls retain
  upstream's default BF16 autocast; weights, state integration, rewards and
  gradient accumulation are FP32. The deterministic equivalence unit test uses
  FP32 network arithmetic to isolate the batching change from BF16 rounding.
- `--eprobe_async 0`: synchronous probe graph instead of 20 side-stream steps
  per training group. The upstream README recommends disabling async probes
  on a shared GPU. Probe sampling/EMA snapshots/focus updates remain enabled.
- All 461 PHP tile probes are swept in fixed batches of 64 worlds with the
  same frozen EMA policy and sampled lead. The final batch is padded and its
  padding discarded; failure counts/focus are applied once after the entire
  sweep. This avoids keeping a second 461-world physics workspace alongside
  training. No clips or tile starts are skipped. This changes execution
  batching, not the probe cadence, coverage, failure window or reset share.
- MJLab 1.6.0, MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.16.0 match upstream's pinned
  runtime. Torch remains 2.13.0+cu130; upstream tested 2.9.0+cu128.
- PHP uses obstacle collisions and 196 relatively short clips rather than
  40 LAFAN1 floor clips. PHP's base task reward/termination differ from the
  LAFAN1 benchmark. These runs do not reproduce its benchmark conditions.
- Physics uses the upstream solver/CCD settings. PHP capacity is 128 contacts
  per world pooled globally and 600 constraints per world. Console guards check
  actual usage and nonfinite transitions; capacities must be increased if a
  run reaches them.
- The enclosing collection/probe CUDA graphs own the physics kernels. Unused
  standalone Warp step/forward graphs are discarded before warmup to avoid
  duplicate graph-executable memory on the 16 GiB GPU. No physics step or
  learning update is removed.

Use `tests/test_blinkmimic_php.py` for adapter contract checks. End-to-end smoke
tests must additionally exercise world-model/policy/collection/probe graphs,
terrain resets and checkpoint evaluation on the installed GPU runtime.
Training wall time and tracking quality must be measured independently before
claiming any speedup on PHP.
