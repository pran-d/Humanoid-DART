<div align="center">

# Humanoid-DART

### **Humanoid** Loco-Manipulation using **D**iffusion-guided **A**ugmentation through **R**elabeling and **T**racking

*IEEE-RAS International Conference on Humanoid Robots (Humanoids), 2026*

<!-- [📄 Paper](#) · [🌐 Project Page](#) · [🎥 Video](#) -->

</div>

<!-- ![Overview](docs/overview.png) -->

> Imitating human demonstrations has become a dominant paradigm for learning
> humanoid loco-manipulation policies, but scaling it is hard: collecting diverse
> demonstrations is costly and policies need continual human correction.
> **Humanoid-DART** is a self-supervised framework that bootstraps from **as few as
> four seed demonstrations** and progressively expands its behavioral repertoire,
> learning a goal-conditioned policy that automatically explores the goal space
> with minimal expert supervision. It couples **diffusion-based trajectory
> generation** with **reinforcement learning**, where RL tracks the
> goal-conditioned trajectories produced by the diffusion model across a range of
> loco-manipulation skills, while a curriculum + goal-relabeling scheme drives
> coverage over a continuous object-goal space.

---

## Overview

Humanoid-DART runs an **evolutionary loop** that alternates between a diffusion
motion generator and a physics-based RL tracker. Each generation produces new
data, which improves both the generator and the policy:

```
                 ┌─────────────────────────────────────────────────────────┐
                 │  evolutionary pipeline  (orchestrator)                    │
                 └─────────────────────────────────────────────────────────┘
 seed motion(s) ─► fit diffusion model ─► train initial RL policy (from scratch)
        │                                                 │
        ▼                                                 ▼
   ┌──────────────────────  per generation  ────────────────────────────────┐
   │  for each iteration:                                                     │
   │   1. sample task-goal targets via a curriculum (frontier / thompson / …) │
   │   2. diffusion model generates N candidate trajectories for those goals  │
   │   3. roll out candidates in MuJoCo-Warp under the current policy         │
   │   4. score them; trajectories beating a fitness threshold become elites  │
   │   5. incrementally fine-tune the diffusion model on the elite set        │
   │  after iterations: fine-tune the RL policy on the accumulated elites     │
   └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   growing archive of elite trajectories + trained policy & diffusion checkpoints
```

The repository is organized around two components:

| Component | Role |
|-----------|------|
| **`mjlab`** ([`src/mjlab/scripts/FoLM/`](mjlab/src/mjlab/scripts/FoLM/)) | MuJoCo-Warp RL simulation, PPO policy training, and the evolutionary loop orchestrator. |
| **`diffusion-planner`** ([`mjlab/src/mjlab/scripts/diffusion_planner/`](mjlab/src/mjlab/scripts/diffusion_planner/)) | Diffusion Transformer that generates full-body G1 trajectories conditioned on task goals. |

---

## Installation

### Prerequisites

- **NVIDIA GPU** with CUDA 12.4+ (required for MuJoCo-Warp training; CPU/macOS is eval-only and very slow)
- **Python 3.10–3.13**
- [`uv`](https://docs.astral.sh/uv/) package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  ```
- A [Weights & Biases](https://wandb.ai) account (logging + optional motion datasets)

### 1. Clone the simulation/RL codebase (`mjlab`)

```bash
git clone https://github.com/Atarilab/mjlab.git
cd mjlab
git checkout user/kanish/FolM
```

### 2. Clone the diffusion planner *into* mjlab

> ⚠️ **Important:** the planner is imported as the Python module
> `mjlab.scripts.diffusion_planner` (**underscore**). The repo clones as
> `diffusion-planner` (hyphen), which is not a valid module name. It **must** be
> placed/renamed as `src/mjlab/scripts/diffusion_planner/`.

```bash
# From inside the mjlab repo root:
git clone https://github.com/pran-d/diffusion-planner.git \
    src/mjlab/scripts/diffusion_planner          # underscore target name
```

If you already cloned it with a hyphen, rename it:
```bash
mv src/mjlab/scripts/diffusion-planner src/mjlab/scripts/diffusion_planner
```

The orchestrator resolves the planner via:
```python
DIFFUSION_PLANNER_PATH = Path(__file__).parent.parent / "diffusion_planner"
from mjlab.scripts.diffusion_planner.motion_generator import MotionGenerator
```
Verify the package markers exist (create empty `__init__.py` if missing):
```bash
ls src/mjlab/scripts/__init__.py \
   src/mjlab/scripts/FoLM/__init__.py \
   src/mjlab/scripts/diffusion_planner/__init__.py
```

### 3. Install the environment

```bash
uv sync                                    # creates .venv, resolves deps from uv.lock
MUJOCO_GL=egl uv run list_envs             # smoke test: prints registered Mjlab-* tasks
```

When run through mjlab, the diffusion model executes inside mjlab's environment,
which already declares the planner's dependencies (`diffusers`, `einops`,
`rotary_embedding_torch`, `timm`, `scipy`, …) — no separate env is needed for the
integrated pipeline.

> *(Optional)* To run the diffusion planner **standalone** (its own `train.py` /
> `inference_mg.py`), install its env separately: `cd src/mjlab/scripts/diffusion_planner && uv sync`.
> See [Standalone diffusion planner](#standalone-diffusion-planner).

### 4. Weights & Biases

```bash
export WANDB_API_KEY=<your-key>            # or: uv run wandb login
export WANDB_ENTITY=<your-org-or-user>
```

---

## Data

The pipeline is seeded by one or more **base motion** `.npz` files. Each holds G1
trajectories with keys such as `base_xyz_quat (T,7)`, `joint_pos (T,29)`,
`obj_xyz_quat (T,7)`; batched files have a leading `(n_batch, T, …)` axis.

Place your motion files where your config points, e.g.
`motions/output/pure_kick/_merged_batched_pure_kick.npz`.

---

## Generating Data / Running the Pipeline

The single entry point is
[`run_evolutionary_pipeline.py`](mjlab/src/mjlab/scripts/FoLM/run_evolutionary_pipeline.py),
driven by [`pipeline_config.yaml`](mjlab/src/mjlab/scripts/FoLM/pipeline_config.yaml).
Any field is overridable on the CLI (**CLI > YAML > default**).

### Minimal run

```bash
MUJOCO_GL=egl uv run python src/mjlab/scripts/FoLM/run_evolutionary_pipeline.py \
  --config src/mjlab/scripts/FoLM/pipeline_config.yaml \
  --base_motion motions/output/pure_kick/_merged_batched_pure_kick.npz \
  --out_dir results/my_first_run \
  --device cuda:0
```

> Always set `MUJOCO_GL=egl` for headless GPU rendering on servers/clusters.
> ⚠️ **`--out_dir` is wiped on start** (`shutil.rmtree`). Use a fresh path each run.

### Warm-starting from prior artifacts

```bash
MUJOCO_GL=egl uv run python src/mjlab/scripts/FoLM/run_evolutionary_pipeline.py \
  --config src/mjlab/scripts/FoLM/pipeline_config.yaml \
  --base_motion motions/.../_merged_batched_stabilized.npz \
  --out_dir results/evo_generation_153 \
  --checkpoint_file logs/rsl_rl/g1_object_tracking/<run>/model_3999.pt \
  --diff_checkpoint results/evo_generation_152/gen_000/diffusion_gen_70.pt \
  --load_elites    results/evo_generation_152/gen_000/elites.json
```

- `--checkpoint_file` — resume the **RL policy** instead of training from scratch
- `--diff_checkpoint` — resume the **diffusion model** weights
- `--load_elites` — preload a previously discovered elite archive

### Key configuration fields

Full schema: [`pipeline_config.py`](mjlab/src/mjlab/scripts/FoLM/pipeline_config.py).

| Field (YAML) | CLI flag | Meaning |
|---|---|---|
| `task` | `--task` | mjlab RL task id, e.g. `Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation` |
| `evo_task` | `--evo_task` | Evolutionary task space (see below) |
| `base_motion` | `--base_motion` | Path to the seed motion `.npz` |
| `out_dir` | `--out_dir` | Output directory (**deleted + recreated** on start) |
| `device` | `--device` | `cuda:0`, etc. |
| `generations` | `--generations` | Outer evolution loops |
| `iterations` | `--iterations` | Inner loops per generation |
| `num_samples` | `--num_samples` / `--num` | Candidate trajectories generated per iteration |
| `curriculum` | `--curriculum` | `random` \| `frontier` \| `bayesian` \| `thompson` |
| `train` | `--train` | Fine-tune the RL policy on elites |
| `training.num_envs` | `--num_envs` | Parallel MuJoCo envs (e.g. 4096 / 8192) |
| `training.initial_max_iterations` | `--initial_max_iterations` | PPO iters for the from-scratch policy |
| `training.elite_max_iterations` | `--elite_max_iterations` | PPO iters per generation fine-tune |
| `diffusion.subset_size` | `--diff_subset_size` | Max elites used to train the diffusion model |
| `diffusion.initial_epochs` | `--diff_initial_epochs` | Diffusion epochs on the base motion |
| `diffusion.train_epochs` | `--diff_train_epochs` | Diffusion epochs per iteration |
| `diffusion.sampling` | `--diff_sampling` | `stratified` \| `clustered` \| `frontier` |
| `diffusion.cfg_weight` | `--diff_cfg_weight` | Classifier-free-guidance strength |
| `evolution.fitness_threshold` | `--fitness_threshold` | Min fitness to qualify as an elite |
| `evolution.max_traj_length` | `--max_traj_length` | Max generated trajectory length |
| `replanning.enabled` | `--replanning` | Closed-loop replanning during rollouts |
| `wandb.project` / `wandb.entity` | `--wandb_project` / `--wandb_entity` | Logging target |

**Available `evo_task` spaces** ([`evo_task_config.py`](mjlab/src/mjlab/scripts/FoLM/evo_task_config.py)):
`box_height`, `pick_place`, `pick_place_final_box_pose`,
`pick_place_relative_box_pose`, `pick_place_relative_box_pose_xyz`,
`pick_place_relative_box_pose_6d`, `pick_place_handout_height`.

### Output artifacts

Inside `results/<out_dir>/`:

```
run_config.yaml                 # resolved config (reproducibility)
diffusion_initial.pt            # diffusion checkpoint after the base-motion fit
gen_000/
  elites.json                   # elite archive metadata (reload via --load_elites)
  elites.npz                    # batched elite trajectories → policy training input
  diffusion_gen.pt              # diffusion checkpoint for this generation
  iter_000/
    batched.npz                 # all generated candidates, batched
    iter_elites.npz             # elites discovered this iteration
    diffusion_raw_outputs.npz   # raw diffusion samples (debug)
    diffusion_pre_rollout.npz   # candidates before MuJoCo rollout
    curriculum_map.png          # sampled goal coverage
    iteration_summary.json      # metrics / timing
  iter_001/ ...
gen_001/ ...
```

RL policy checkpoints land under `logs/rsl_rl/<experiment_name>/<run>/model_*.pt`.

---

## Training a Policy Directly

To train a tracking policy on a motion without the evolutionary loop:

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation \
  --motion-file motions/output/motion.npz \
  --env.scene.num_envs 4096 --agent.max-iterations 30000 --device cuda:0
```

### Sanity-checking a task before training

```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation --agent zero
uv run play Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation --agent random
```

---

## Standalone Diffusion Planner

For training/debugging the generator in isolation. Run from inside
`src/mjlab/scripts/diffusion_planner/` using **its own** environment. First point
`config/paths.yaml` at your data/checkpoint/log dirs; main settings live in
`config/config.yaml`.

```bash
cd src/mjlab/scripts/diffusion_planner

# Train
uv run python train.py                       # from scratch
uv run python train.py --resume 500          # resume from epoch 500

# Inference (uses config/inference.yaml defaults)
MUJOCO_GL=egl uv run python inference_mg.py --epoch 500 --ema --stitch_steps 10
MUJOCO_GL=egl uv run python inference_mg.py --epoch 500 --task_params 0.5 -0.3 --cfg_w 1.5
#   --style pick|push|kick   --save_path results/run.npz

# Goal sweep / visualisation
MUJOCO_GL=egl uv run python batch_goal_sweep.py --epoch 500 --traj_idx 0 --num_goals 6
uv run python plot_trajectory_npz.py --npz_path results/inference_mg.npz
```

The planner outputs world-frame trajectories of shape `(B, T, 43)` (pelvis
xyz+quat + 29 joints + object xyz+quat). See the planner's `README.md`,
`CLAUDE.md`, and the `docs/` folder for the 51-dim ego-centric feature layout,
SBTO transform, style conditioning, and hierarchical two-phase generation.

---

## Quick-Start Checklist

1. Install `uv`; have an NVIDIA GPU (CUDA 12.4+).
2. `git clone mjlab` → `git checkout user/kanish/FolM`.
3. `git clone diffusion-planner` into `src/mjlab/scripts/diffusion_planner/` (**underscore!**).
4. `uv sync`; verify `uv run list_envs`.
5. `wandb login` / set `WANDB_API_KEY` + `WANDB_ENTITY`.
6. Put a base motion `.npz` under `motions/…`.
7. Edit `pipeline_config.yaml` (`task`, `evo_task`, `base_motion`).
8. Run the pipeline (see [above](#minimal-run)).
9. Collect data from `results/<run>/gen_*/elites.npz` and policy checkpoints from `logs/rsl_rl/.../model_*.pt`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: mjlab.scripts.diffusion_planner` | Directory is named with a hyphen — rename to `diffusion_planner`; ensure `__init__.py` files exist. |
| OpenGL / EGL / rendering errors on a server | Prefix commands with `MUJOCO_GL=egl`. |
| Out-of-memory during training | Lower `--num_envs` (8192 → 4096 → 2048) and/or `--diff_chunk_size`. |
| Previous results disappeared | Expected — `--out_dir` is `rmtree`'d at start. Use a new path. |
| WandB prompts / fails | `export WANDB_API_KEY=...` and `WANDB_ENTITY=...`. |

---

## Citation

If you find Humanoid-DART useful, please cite:

```bibtex
@inproceedings{humanoiddart2026,
  title     = {Humanoid-DART: Humanoid Loco-Manipulation using Diffusion-guided
               Augmentation through Relabeling and Tracking},
  author    = {Anonymous Author(s)},
  booktitle = {IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year      = {2026}
}
```

## Acknowledgments

Humanoid-DART builds on [mjlab](https://github.com/Atarilab/mjlab) (Isaac Lab API
powered by MuJoCo-Warp) and the [diffusion-planner](https://github.com/pran-d/diffusion-planner).
We thank the Isaac Lab and MuJoCo-Warp teams for their open-source work.

## License

See the `LICENSE` files in the respective component repositories
([mjlab](mjlab/LICENSE) is Apache-2.0).
