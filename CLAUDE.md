# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CLoSD (ICLR 2025) closes the loop between a real-time diffusion motion planner ("DiP", built on MDM)
and a physics-based imitation controller (PHC, built on IsaacGym + rl_games) to give a physically
simulated humanoid multi-task control from text prompts. DiP proposes short motion plans in HumanML3D
representation; a low-level RL tracking controller executes them in simulation; the resulting simulated
pose is fed back into DiP as context for the next planning step — hence "closing the loop."

## Environment setup

- Requires Python 3.8.19 (tested on Ubuntu 20.04.5), a single GPU (~4GB for running, ~50GB for
  train/eval), and IsaacGym (NVIDIA proprietary, must be downloaded separately and pip-installed
  into the conda env — it is not in `requirement.txt`).

```
conda create -n closd python=3.8
conda activate closd
pip install -r requirement.txt
python -m spacy download en_core_web_sm
# then install Isaac Gym into the same env from <ISAAC_GYM_DIR>/python
```

- There is no separate data/checkpoint setup step: `closd/utils/hf_handler.py` downloads/caches
  everything from the `guytevet/CLoSD` HuggingFace repo on first run and symlinks DiP checkpoints,
  evaluation data, and CLoSD checkpoints into the expected local paths.
- SMPL/AMASS/HumanML3D licenses must be individually honored (data auto-downloaded but license terms
  are the user's responsibility).

## Running things

Everything runs through `closd/run.py`, a Hydra entry point (config root: `closd/data/cfg`,
top-level config `config.yaml`). All CLI invocations are of the form:

```
python closd/run.py learning=<...> robot=<...> env=<...> exp_name=<...> [key=value ...]
```

Key env variants (`closd/data/cfg/env/*.yaml`, selects the task class via `parse_task.py`'s
`eval(args.task)(...)`):
- `env=closd_multitask` → task `CLoSDMultiTask` — multiple task states (strike/sit/reach/etc.)
- `env=closd_sequence` → task `CLoSDSequence` — runs a scripted sequence of tasks
- `env=closd_t2m` → task `CLoSDT2M` — pure text-to-motion
- `env=im_single_prim` → low-level tracking-only training (no DiP loop)

Common flags: `test=True` (inference/play mode), `headless=True/False` (monitor on/off,
`no_virtual_display=True` needed when running with a monitor), `epoch=-1` (resume last checkpoint),
`env.num_envs=N`, `exp_name=CLoSD_no_finetune` vs `CLoSD_multitask_finetune`/`CLoSD_t2m_finetune`
(fine-tuned vs not), `env.dip.cfg_param=<classifier-free guidance scale>`, `learning=im_toy` +
`no_log=True env.num_envs=4` for a fast debug run. For `env=closd_t2m`, `env.dip.custom_prompt=<text>`
overrides the dataset-sampled prompt so every env acts out the given text instead
(`closd_t2m.py`'s `CLoSDT2M.update_mdm_conditions`); `./scripts/run_prompt.sh "<prompt text>"` wraps
this into a single command (`--dry-run` to print without executing, `-h` for other options).
For a *sequence* of prompts played back-to-back in one continuous rollout (e.g. squat then clap),
`env.dip.custom_prompt_sequence=<a|b|c>` + `env.dip.custom_prompt_sequence_frames=<n1|n2|n3>`
(pipe-separated; frames default to `env.dip.default_segment_frames`) swap `hml_prompts` mid-episode
via `CLoSDT2M.update_state_machine` — the pose buffer carries the real simulated pose forward, so
each prompt plans onward from where the previous one left the character (no explicit stitching).
`./scripts/run_prompt_sequence.sh "<prompt 1>" "<prompt 2>" [...]` wraps this
(`-s/--seconds-per-prompt`, `--dry-run`, `-h`). Design notes / rationale for these custom-prompt
features live in `implemented_plans/` (`custom_prompt.md`, `run_prompt_script.md`) — read them
before extending the prompt/sequence machinery, as they document what is deliberately out of scope
(e.g. no language→3D-target grounding for `closd_t2m`).

Standalone DiP (the diffusion planner) can also be sampled/evaluated/trained without the CLoSD/IsaacGym
loop via `python -m closd.diffusion_planner.<sample.generate|eval.eval_humanml|train.train_mdm>`
— see README.md for exact invocations of multi-task run/eval/train, DiP standalone sampling, and
Blender/SMPL visualization/extraction workflows. There is no test suite, linter, or CI config in
this repo — validate changes by running the relevant `closd/run.py` invocation or DiP standalone
script above.

## Architecture

### Task class hierarchy (`closd/env/tasks/`)

IsaacGym task classes build on each other, each adding a layer of behavior. Understanding a bug
usually means walking this chain:

```
Humanoid (base physics/character sim, ASE/PHC-derived)
  → HumanoidAMP (adds AMP observation buffers)
    → HumanoidIm (adds motion imitation: reference-motion matching, rewards, motion lib caching)
      → CLoSD (closd.py) — wires in DiP: runs the diffusion planner every `planning_horizon` sim
                            steps, converts sim pose ↔ HumanML3D representation via
                            `RepresentationHandler` (closd/utils/rep_util.py), maintains a rolling
                            `pose_buffer` used as DiP's conditioning context, and (optionally)
                            records rollouts back into HumanML3D format for evaluation
                            (`save_hml_episodes`).
        → CLoSDTask (closd_task.py) — adds the finite state machine (`STATES` enum in
                      closd_util.py: REACH, STRIKE_KICK, STRIKE_PUNCH, HALT, SIT, GET_UP,
                      TEXT2MOTION) that drives which text prompt / target joints DiP is
                      conditioned on (`state_machine_conditions`), and done-detection based on
                      distance to a target.
          → CLoSDT2M (closd_t2m.py) — pure text-to-motion, no state machine task logic.
          → CLoSDMultiTask (closd_multitask.py) — randomized per-env task assignment
                            (strike/sit/reach/etc.), task-specific target placement and rewards.
          → CLoSDSequence (closd_sequence.py) — drives a fixed ordered sequence of states per
                            episode instead of one random task.
```

`parse_task.py` instantiates the task class named by `cfg.env.task` (a plain string, `eval()`'d)
and wraps it in `VecTaskPythonWrapper` for rl_games. `closd/run.py` registers the rl_games
env/vecenv and algo/player/model/network builders (`amp`, `im_amp` — the imitation+AMP agent used
for training/fine-tuning the tracking controller) before handing off to `rl_games.Runner`.

### DiP / diffusion planner (`closd/diffusion_planner/`)

A largely self-contained MDM-derived (motion-diffusion-model) package: `model/` (transformer
diffusion model, BERT text encoder), `diffusion/` (diffusion process/sampler), `data_loaders/humanml/`
(HumanML3D dataset + representation utilities, kinematic chain, recover-from-RIC, plotting),
`sample/generate.py` (standalone sampling), `eval/eval_humanml.py` (standalone HumanML3D metrics),
`train/train_mdm.py` (standalone DiP training). CLoSD's `closd.py` calls into this package directly
(`create_model_and_diffusion`, `sample_fn = diffusion.p_sample_loop`, `ClassifierFreeSampleModel`)
rather than through a CLI — DiP checkpoints are plain `.pt` files loaded via `load_saved_model`.

### Learning / RL agents (`closd/learning/`)

rl_games-based PPO/AMP agents: `amp_agent.py`/`amp_players.py`/`amp_models.py`/`amp_network_builder.py`
implement adversarial motion prior (AMP) training; `im_amp.py`/`im_amp_players.py` extend these for
imitation-with-AMP (the tracking controller used inside CLoSD). Configs for these live in
`closd/data/cfg/learning/*.yaml` (network/algorithm hyperparams: `im`, `im_big`, `im_toy`) and
`closd/data/cfg/train/rlg/*.yaml` (rl_games trainer configs, incl. PNN/MCP variants).

### Representation conversion (`closd/utils/rep_util.py`, `closd/utils/motion_lib_*.py`)

The SMPL/mujoco skeleton (24 joints, used by the physics sim) and the HumanML3D 22-joint
representation (used by DiP) are different orderings/formats; `mujoco_2_smpl`/`smpl_2_mujoco` index
arrays and `RepresentationHandler.pose_to_hml`/`hml_to_pose` handle the conversion each planning
step. `closd/utils/poselib/` (vendored) provides skeleton/motion tensor utilities used in this
conversion (e.g. `SkeletonMotion._compute_velocity`).

### Config composition (Hydra, `closd/data/cfg/`)

`config.yaml` composes defaults from `env/`, `robot/`, `learning/`, `sim/` groups. Env configs
inherit via Hydra `defaults:` (e.g. `closd_t2m.yaml` extends `closd_base.yaml` which extends
`im_single_prim.yaml`) — when tracing a config value, check the whole chain, not just the leaf
file. Because of this inheritance, code that reads a nested key which only exists in some leaf
configs (e.g. `cfg['env']['save_motion']`) must use `.get(...)` with a default rather than direct
indexing, or it will crash for env configs that don't define that key (see the `init_save_hml_episodes`
fix in git history for a concrete example of this class of bug).

### Output/state

Trained checkpoints and logs go to `output/<project>/<exp_name>/`. IsaacGym viewer recordings
(press `L` during a run) save to `output/states/`, which can be converted to SMPL params via
`closd/utils/extract_smpl.py` or visualized in Blender via `closd/blender/record2anim.py`.
