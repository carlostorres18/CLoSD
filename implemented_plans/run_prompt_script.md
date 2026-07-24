# Plan: Wrapper script for the custom-prompt CLoSD demo

## Goal

A single script the user can run with just a prompt string, instead of having to
remember/retype the full `closd/run.py` Hydra invocation every time, e.g.:

```
./scripts/run_prompt.sh "a person does a cartwheel"
```

This builds directly on [implemented_plans/custom_prompt.md](implemented_plans/custom_prompt.md)
(`env.dip.custom_prompt`) — the script is just a thin, memorable wrapper around the command
documented there.

## Location & naming

- New `scripts/` directory at repo root (doesn't exist yet — first script in the repo).
- `scripts/run_prompt.sh`, marked executable (`chmod +x`).

## Interface

```
./scripts/run_prompt.sh "<prompt text>" [options]

Options:
  -n, --num-envs N     number of parallel simulated envs (default: 9)
  --no-finetune        use exp_name=CLoSD_no_finetune instead of CLoSD_t2m_finetune
  --headless           run without the viewer window (headless=True, drops no_virtual_display)
  --dry-run            print the python command that would run, but don't execute it
  -h, --help           usage message
```

Only the prompt is required; everything else defaults to matching the command we've been
running manually (`learning=im_big robot=smpl_humanoid epoch=-1 test=True
no_virtual_display=True headless=False env=closd_t2m exp_name=CLoSD_t2m_finetune
env.num_envs=9`).

## Implementation details

1. **Argument parsing**: first positional arg is the prompt (required, quoted by the user
   so it can contain spaces/punctuation); remaining args parsed with a small `getopts`/manual
   loop for the options above.
2. **Sanity checks before launching** (fail fast with a clear message instead of a deep
   Python/Hydra traceback):
   - Prompt was actually provided → else print usage and exit 1.
   - `$CONDA_DEFAULT_ENV` equals `closd` → else warn "conda env 'closd' is not active; run
     `conda activate closd` first" and exit 1 (catches the exact `libpython3.8.so` failure
     mode from earlier in this project).
   - `nvidia-smi` runs successfully → else warn "GPU driver not available (nvidia-smi
     failed) — check for a driver/library version mismatch" and exit 1 (catches the driver
     mismatch issue we hit earlier, instead of letting it surface as a CUDA error deep in a
     rl_games stack trace).
3. **Build the command** as a bash array (avoids quoting bugs), substituting the parsed
   options:
   ```bash
   cmd=(python closd/run.py
     learning=im_big robot=smpl_humanoid
     epoch=-1 test=True
     env.num_envs="$num_envs"
     env=closd_t2m exp_name="$exp_name"
     "env.dip.custom_prompt=$prompt"
   )
   if [ "$headless" = true ]; then
     cmd+=(headless=True)
   else
     cmd+=(headless=False no_virtual_display=True)
   fi
   ```
4. **`--dry-run`**: print the assembled command (properly quoted) and exit without running
   it — useful for verifying the prompt/flags are being passed the way the user expects
   before committing to a full IsaacGym launch.
5. **Run from repo root**: the script should `cd` to the repo root (directory containing
   `closd/`) before invoking `python closd/run.py`, so it works regardless of the caller's
   current directory — resolve this via the script's own path
   (`cd "$(dirname "${BASH_SOURCE[0]}")/.."`).

## Edge cases / considerations

- **Quoting the prompt through to Hydra**: building the override as a single array element
  (`"env.dip.custom_prompt=$prompt"`) avoids word-splitting issues that a naive
  `env.dip.custom_prompt=$prompt` (unquoted) would hit with multi-word prompts.
- **Prompts containing `=` or `,`**: Hydra's override parser can be picky about these
  characters inside a value; note this as a known limitation in the script's `--help` text
  rather than trying to fully solve escaping for every edge case up front.
- **Not duplicating the LD_LIBRARY_PATH fix**: that's already handled by the conda
  `activate.d` hook set up earlier, so the script doesn't need to touch it — just checks
  the env is active.
- **Keep it a thin wrapper**: no new Python code, no new config beyond what
  `custom_prompt.md` already added — this script only assembles and runs the CLI command.

## Testing / validation

1. `--dry-run` with a simple prompt → confirm the printed command matches the manual
   invocation used previously (same flags/defaults).
2. `--dry-run` with a multi-word, punctuated prompt (e.g. `"a person waves, then sits down."`)
   → confirm it's captured as a single value.
3. Run without activating the conda env → confirm the script exits early with the
   friendly conda-env message instead of an import traceback.
4. Run with a real prompt end-to-end once the GPU is healthy → confirm it launches the
   same way the manual command did and visualizes the prompt in the viewer.
5. Run with `--no-finetune` and `--num-envs 1` → confirm those options correctly override
   the defaults in the assembled command.

## Out of scope

- No support yet for switching to `closd_multitask`/`closd_sequence` or other env variants
  — this script is specifically the custom-prompt T2M demo. A more general "any command"
  wrapper could be a future follow-up if useful.
