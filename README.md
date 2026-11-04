# PHP Parkour with BlinkMimic

PHP adaptation of [BlinkMimic](https://github.com/fuyichen0725/blinkmimic), pinned at
`d264658d6c9ce61fb52f9f0f8985a369f3ccc409`. The default training entry point now
uses the complete released `PRESET[main]` method. The earlier execution-only
`--fast` implementation remains in `supermimic_parkour` as legacy code.

The port and explicit runtime/task differences are documented in
[blinkmimic/PHP_PORT.md](blinkmimic/PHP_PORT.md). Upstream's approximately
3-minute result was on 40 LAFAN1 clips on an otherwise idle RTX 4090; it is not
a measured result for this PHP obstacle task.

## Setup and data

```bash
conda activate mjlab
python -m pip install -e '.[viewer]'
```

Runtime pins: MJLab 1.6.0, MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.16.0.
The local RTX 5080 environment uses Torch 2.13.0+cu130; upstream tested Torch
2.9.0+cu128. Data is external and is not distributed in this repository.

Put the [PHP motion bundles](https://github.com/amazon-far/php_parkour/tree/main/motion_matching)
under `data/php_motion`, or pass `--dataset-dir /path/to/php_motion`:

```text
php_motion/
  php-motions-low-step-v1/*_motion.npz, *_terrain.npy
  php-motions-high-step-v1/*_motion.npz, *_terrain.npy
  php-motions-low-climb-76-v1/*_motion.npz, *_terrain.npy
  php-motions-high-climb-76-v1/*_motion.npz, *_terrain.npy
```

The local four-skill library has 196 clips and 110194 frames at 50 Hz. Ultra-Climb
is absent and is not silently substituted. Named joints/bodies, floating-base
prefixes, sampling bins, terrain variants and quaternions are validated by the
PHP loader. Checkpoints record hashes of all motion/terrain pairs.

## Fresh training: 2048 environments

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1 \
python scripts/train_php_blinkmimic.py \
  --dataset-dir data/php_motion --num_envs 2048 \
  --log_dir logs/blinkmimic_php_fresh
```

`scripts/train_php_supermimic.py` and `supermimic-train` also use this recipe.
Omit `--resume` to initialize new weights, optimizers, statistics and replay data.
Use a new log directory. Defaults: 3500 update groups; each group fits the world
model five times and updates the policy/critic once. With UTD 0.4, a group collects
2 or 3 vectorized physics steps. These iteration numbers do not equal legacy
training's vector-step counter.

The launcher retains the upstream 2048 policy and world-model batches, 32-step
imagined horizon, Muon, EMA, structured dynamics, reference-action skip,
mirroring, contact reward, failure/evaluation-guided resets, learning-rate
schedule and all training/collection/probe CUDA graphs. Local hardware overrides
are 2048 simulator environments, replay length 512, and synchronous probes
(`--eprobe_async 0`, upstream's recommendation for a shared GPU). First-time
Torch/Warp compilation can take several minutes.

On the 16 GiB GPU, actor BPTT uses four 512-trajectory chunks, with weighted
FP32 gradient accumulation and one optimizer/EMA update over all 2048
trajectories (`--pi_microbatch 512`). Critic normalization and its update schedule
still use the full batch. This reduces activation memory without shrinking
the logical batch; numerical accumulation order and noise draws differ.
Probe sweeps still cover all 461 PHP tiles, executed in batches of 64 physical
worlds with a frozen EMA policy to limit additional physics workspace memory.
Network matrix products retain upstream's default BF16 autocast; weights,
physics states, integration, rewards and accumulated gradients stay FP32.

`args.json` records the actual arguments and data contract; `train_log.csv`
records learning metrics and elapsed time. Periodic `ckpt_*.pt` files contain
the deployed EMA policy every 100 groups; the final `last.pt` also includes
optimizer/world-model/control state for a warm resume. Mid-run light policy
checkpoints cannot resume training. Console physics checks abort on nonfinite
transitions or exhausted contact/constraint capacity.

## Evaluation and playback

```bash
python scripts/evaluate_php_supermimic.py \
  --checkpoint logs/blinkmimic_php_fresh/ckpt_000100.pt \
  --dataset-dir data/php_motion

python scripts/play_php_supermimic.py \
  --checkpoint logs/blinkmimic_php_fresh/ckpt_000100.pt \
  --dataset-dir data/php_motion --device cuda:0 --port 8082
```

Evaluation starts every training clip at its first frame, with terrain variant
0 and no randomization/noise, and saves normalized return/failure counts plus
per-skill results to JSON. This measures reference-conditioned motion tracking;
it does not establish autonomous visual obstacle traversal or hardware safety.
Playback offers motion/terrain selection and a reference ghost at
`http://127.0.0.1:8082`.

The legacy modules remain accessible through `supermimic-train-legacy`,
`supermimic-play-legacy`, and `supermimic-evaluate-legacy`. Their checkpoint
architecture differs from the new PHP BlinkMimic format.

## PHP 619-motion collection: 100000 update groups

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1 \
python scripts/train_php619_blinkmimic.py --log_dir logs/php619_fresh_100k
```

The launcher reads the explicitly indexed collection at
`/home/ljc/php_parkour/experiments/supermimic_parkour_isaacsim/data/php_full17_merged619_20261104`
(override with `--dataset-dir`). Its 619 active clips have 336637 frames across
17 skills: 69 geometry-repaired references and 550 original references retaining
their quality flags. The 10 quarantined clips are excluded. All indexed motion
and terrain hashes are checked; source files are not modified. The two cohorts'
named states/velocities and per-clip provenance are retained. This is an
experimental training run, not certification of the unrepaired references.

The mesh adapter preserves the source triangle surfaces, including concavities,
using twelve convex tetrahedra per star-shaped obstacle. It validates closed
topology and decomposition before building the collision assets. Up to 15
obstacles are represented. Network context contains the actual eight vertices
per obstacle; its geometry lookup ID is excluded from network inputs. Foot
support heights are computed on the source triangles. Compound contact
discretization is MuJoCo-specific; this does not assert PhysX trajectory parity.

The long-run configuration uses 2048 physical worlds, logical batches of 2048,
128-trajectory actor chunks, and replay length 128 to fit the mesh workload on
the shared 16 GiB GPU. The learning-rate schedule spans 100000 update groups.
Evaluation-guided sweeps cover all 1493 tiles every 1000 groups in batches of
32 worlds; the failure-reset mechanisms remain enabled. A quarter of resets
start at clip entry, using the collection's skill-balanced clip distribution.

The launcher freezes the active motion/terrain files and their cohort metadata
under the run's `dataset/`, checks all 1238 active file hashes, and copies the
training code/assets to `source/`. Training, recovery and the final evaluation
use those snapshots. `input_snapshots.json` records their provenance and source
hashes; quarantined clips and checkpoint archives are not copied.

The first update and each 1000-group interval save a full atomic `last.pt`, in
addition to milestone checkpoints. SIGTERM/SIGINT finish the current group and
save full recovery state. Resume the same dataset/run with:

```bash
python scripts/train_php619_blinkmimic.py \
  --resume logs/php619_fresh_100k/last.pt --log_dir logs/php619_fresh_100k
```

Resume restores networks, optimizers, EMA and sampler/schedule state; simulation
and replay are refilled as in upstream. Successful completion automatically
evaluates the final policy on every complete active clip, terrain variant 0,
and writes `final_evaluation_variant0.json` (`--no-final-eval` skips that step).
