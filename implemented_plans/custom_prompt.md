# Plan: Custom text prompt for CLoSD (`env=closd_t2m`)

## Goal

Let a user type their own free-text action prompt (e.g. `"a person does a cartwheel"`,
`"walk forward"`) on the command line and have the full CLoSD closed-loop (DiP planner +
physics tracking controller) act it out, visualized live in the IsaacGym viewer.

**Out of scope:** prompts that reference a specific object/location in the scene
(e.g. `"go to the couch"`). CLoSD has no language→3D-location grounding today; the
multi-target DiP checkpoint used by `closd_multitask`/`closd_sequence` gets its target
coordinates from hand-written task logic (e.g. bench placement), not from parsing text.
Adding that is a separate, larger feature and not addressed by this plan.

## Why this is feasible with a small change

`env=closd_t2m` already runs the full closed-loop simulation and renders it in the
IsaacGym viewer. The only gap: `CLoSDT2M.update_mdm_conditions()`
([closd/env/tasks/closd_t2m.py:47-65](closd/env/tasks/closd_t2m.py#L47-L65)) currently
overwrites `self.hml_prompts[i]` every episode reset with a random sample pulled from the
HumanML3D test set via `self.mdm_data_iter`. Nothing downstream cares *where* the string in
`self.hml_prompts` came from — `get_text_prompts()`
([closd/env/tasks/closd.py:285-289](closd/env/tasks/closd.py#L285-L289)) just returns
whatever is currently in that list, encodes it, and feeds it to DiP as text conditioning
every `planning_horizon` steps. The DiP checkpoint used for `closd_t2m`
(`DiP_no-target_10steps_context20_predict40`) has no target-joint conditioning
(`multi_target_cond=False`), so it's designed for exactly this kind of free-text-only prompt.

## Implementation steps

1. **Add a config field** for the custom prompt, defaulting to empty (meaning: fall back to
   current dataset-sampling behavior). Add to
   [closd/data/cfg/env/closd_t2m.yaml](closd/data/cfg/env/closd_t2m.yaml) under `dip:`:
   ```yaml
   dip:
     custom_prompt: ''   # if non-empty, used as the text prompt for every env instead of sampling from the dataset
   ```

2. **Modify `CLoSDT2M.update_mdm_conditions()`** in `closd/env/tasks/closd_t2m.py`:
   - Read `self.cfg['env']['dip'].get('custom_prompt', '')` once (e.g. cache it in `__init__`
     as `self.custom_prompt`, following the existing `.get(...)`-with-default pattern used
     elsewhere in this codebase for leaf-config keys — see `init_save_hml_episodes` in
     `closd.py` for precedent).
   - If `self.custom_prompt` is non-empty:
     - Set `self.hml_prompts[int(i)] = self.custom_prompt` for every `i` in `env_ids`,
       instead of reading `model_kwargs['y']['text'][int(i)]`.
     - Skip assigning `self.hml_lengths`/`self.hml_tokens`/`self.db_keys`/
       `self.hml_prefix_from_data` for these envs (they're only used for HumanML3D
       evaluation bookkeeping and for seeding the first couple of planning steps from a
       real dataset motion snippet — unnecessary here since the physical character already
       initializes its pose buffer from its own real standing pose in `_reset_pose_buffer`,
       the same way `closd_multitask` operates with no dataset prefix at all).
     - Still call `next(self.mdm_data_iter)` only if you still need `gt_motion` for other
       envs running without a custom prompt in the same batch — simplest first version:
       when `custom_prompt` is set, skip the dataset iterator entirely for all envs.
   - Leave the dataset-sampling path completely unchanged when `custom_prompt` is empty, so
     default behavior (and existing evaluation runs) are unaffected.

3. **No changes needed** in `closd.py`, `closd_task.py`, or the DiP model/sampler — they
   already just consume `self.hml_prompts` generically.

## Usage once implemented

```
python closd/run.py \
  learning=im_big robot=smpl_humanoid \
  epoch=-1 test=True no_virtual_display=True \
  headless=False env.num_envs=9 \
  env=closd_t2m exp_name=CLoSD_t2m_finetune \
  env.dip.custom_prompt="a person does a cartwheel"
```

With `env.num_envs=9`, all 9 simulated humanoids in the viewer will perform the same
prompt simultaneously (useful for comparing multiple random seeds of one action at once).

## Edge cases / considerations

- **Quoting on the CLI:** multi-word prompts need quotes (`env.dip.custom_prompt="..."`);
  Hydra should pass the full quoted string through as one value — verify this works with a
  prompt containing a comma or equals sign, which Hydra's override syntax can be picky about.
- **`get_cur_done()`** for `CLoSDT2M` already always returns `False`
  ([closd_t2m.py:67-69](closd/env/tasks/closd_t2m.py#L67-L69)), so episodes just run for
  `episode_length` steps and then reset — a custom prompt will keep being reused across
  resets since it's re-applied in `update_mdm_conditions()` every reset, not just once.
- **`save_hml_episodes`** (recording rollouts back to HumanML3D format for evaluation) should
  probably be left disabled when using a custom prompt — recorded episodes would all share
  the same caption, which is fine for casual use but meaningless for benchmark evaluation.
- **Debug visualization:** `env.dip.debug_hml=True` already prints/saves the prompt in use
  ([closd.py:330-336](closd/env/tasks/closd.py#L330-L336)) — useful to sanity-check the
  custom prompt is actually being picked up.

## Testing / validation

1. Run with `env.dip.custom_prompt=""` (default) and confirm behavior is unchanged
   (prompts still sampled from the dataset).
2. Run with a simple custom prompt (`"a person walks forward"`) at `env.num_envs=1
   headless=False no_virtual_display=True` and visually confirm the character performs
   the described action in the IsaacGym viewer.
3. Try a couple of distinct action prompts (e.g. `"a person jumps"`, `"a person waves with
   the right hand"`) to confirm the text conditioning is actually influencing motion, not
   just falling through to some default.
4. Confirm `env.num_envs=9` broadcasts the same prompt to all 9 environments correctly.

## Possible future extension (not in this plan)

Spatial-goal prompts (e.g. "go to the couch") would require: (a) detecting/placing the
referenced object in the scene, (b) picking a target joint (e.g. pelvis) and 3D coordinates
for it, (c) generating or selecting a matching text string, and (d) switching to the
multi-target DiP checkpoint (`multi_target_cond=True`). This is a materially larger feature
and would build on top of the task-specific target logic already used by
`closd_multitask.py`'s `SIT`/`REACH` states rather than on `closd_t2m`.
