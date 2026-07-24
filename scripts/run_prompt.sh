#!/usr/bin/env bash
#
# run_prompt.sh — run the CLoSD text-to-motion demo with a custom prompt.
#
# Thin wrapper around `closd/run.py` (env=closd_t2m) that feeds your own text to the
# DiP planner via env.dip.custom_prompt (see implemented_plans/custom_prompt.md).
#
# Usage:
#   ./scripts/run_prompt.sh "<prompt text>" [options]
#
# Example:
#   ./scripts/run_prompt.sh "a person does a cartwheel"
#
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/run_prompt.sh "<prompt text>" [options]

Run the CLoSD text-to-motion demo, having the simulated humanoid perform your
own text prompt (visualized live in the IsaacGym viewer).

Arguments:
  <prompt text>        Required. Free-text action, e.g. "a person does a cartwheel".
                       Quote it so spaces/punctuation are kept as a single value.

Options:
  -n, --num-envs N     Number of parallel simulated envs (default: 9).
      --no-finetune    Use exp_name=CLoSD_no_finetune instead of CLoSD_t2m_finetune.
      --headless       Run without the viewer window (headless=True).
      --dry-run        Print the python command that would run, but don't execute it.
  -h, --help           Show this message.

Examples:
  ./scripts/run_prompt.sh "a person walks forward"
  ./scripts/run_prompt.sh "a person jumps" --num-envs 1
  ./scripts/run_prompt.sh "a person waves" --headless --dry-run

Notes:
  * Requires the 'closd' conda env to be active (conda activate closd) and a
    working GPU driver (nvidia-smi).
  * Hydra's override parser can be picky about prompts containing '=' or ','.
    Simple action phrases work best.
EOF
}

# ---- defaults ----------------------------------------------------------------
prompt=""
num_envs=9
exp_name="CLoSD_t2m_finetune"
headless=false
dry_run=false

# ---- argument parsing --------------------------------------------------------
# First non-option argument is the prompt; the rest are options.
positional=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--num-envs)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a value." >&2
                exit 1
            fi
            num_envs="$2"
            shift 2
            ;;
        --no-finetune)
            exp_name="CLoSD_no_finetune"
            shift
            ;;
        --headless)
            headless=true
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --)
            shift
            while [ $# -gt 0 ]; do positional+=("$1"); shift; done
            ;;
        -*)
            echo "Error: unknown option '$1'." >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if [ "${#positional[@]}" -gt 0 ]; then
    prompt="${positional[0]}"
fi
if [ "${#positional[@]}" -gt 1 ]; then
    echo "Error: multiple prompt arguments given: ${positional[*]}" >&2
    echo "Wrap your prompt in quotes, e.g. \"a person does a cartwheel\"." >&2
    exit 1
fi

# ---- sanity checks -----------------------------------------------------------
if [ -z "$prompt" ]; then
    echo "Error: no prompt provided." >&2
    echo >&2
    usage >&2
    exit 1
fi

# Validate num_envs is a positive integer.
if ! [[ "$num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-envs must be a positive integer (got '$num_envs')." >&2
    exit 1
fi

# Checks that would only fail at launch are skipped for --dry-run, so you can
# inspect the command without a GPU or the conda env active.
if [ "$dry_run" = false ]; then
    if [ "${CONDA_DEFAULT_ENV:-}" != "closd" ]; then
        echo "Error: conda env 'closd' is not active (current: '${CONDA_DEFAULT_ENV:-none}')." >&2
        echo "Run 'conda activate closd' first, then re-run this script." >&2
        exit 1
    fi

    if ! nvidia-smi >/dev/null 2>&1; then
        echo "Error: GPU driver not available (nvidia-smi failed)." >&2
        echo "Check for an NVIDIA driver/library version mismatch (a reboot often fixes it)." >&2
        exit 1
    fi
fi

# ---- run from repo root ------------------------------------------------------
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# ---- build the command -------------------------------------------------------
cmd=(python closd/run.py
    learning=im_big robot=smpl_humanoid
    epoch=-1 test=True
    env.num_envs="$num_envs"
    env=closd_t2m "exp_name=$exp_name"
    "env.dip.custom_prompt=$prompt"
)
if [ "$headless" = true ]; then
    cmd+=(headless=True)
else
    cmd+=(headless=False no_virtual_display=True)
fi

# ---- dry-run or execute ------------------------------------------------------
if [ "$dry_run" = true ]; then
    echo "Would run (from $repo_root):"
    # Print each token quoted so the prompt is unambiguous.
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

echo "Running CLoSD with prompt: \"$prompt\" (num_envs=$num_envs, exp_name=$exp_name)"
exec "${cmd[@]}"
