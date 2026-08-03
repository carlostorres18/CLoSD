#!/usr/bin/env bash
#
# run_prompt_sequence.sh — run the CLoSD text-to-motion demo with a SEQUENCE of prompts.
#
# Thin wrapper around `closd/run.py` (env=closd_t2m) that feeds DiP an ordered list of your
# own prompts (e.g. "squat" then "clap"). Each prompt plays for a fixed duration, then the
# next one is swapped in mid-episode — the humanoid plans the next action starting from
# wherever it physically ended up (see implemented_plans/custom_prompt.md and
# env.dip.custom_prompt_sequence in closd/data/cfg/env/closd_t2m.yaml).
#
# Usage:
#   ./scripts/run_prompt_sequence.sh "<prompt 1>" "<prompt 2>" [...] [options]
#
# Example:
#   ./scripts/run_prompt_sequence.sh "a person squats down" "a person claps"
#
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/run_prompt_sequence.sh "<prompt 1>" "<prompt 2>" [...] [options]

Run the CLoSD text-to-motion demo, having the simulated humanoid perform a sequence of
your own text prompts one after another (visualized live in the IsaacGym viewer). Each
prompt is planned starting from the character's real pose at the end of the previous one.

Arguments:
  <prompt N>           One or more free-text actions, each quoted separately, e.g.
                       "a person squats down" "a person claps".

Options:
  -s, --seconds-per-prompt N   Seconds each prompt plays before switching (default: 3).
                               Converted to frames at 30fps (N*30) per prompt.
  -n, --num-envs N     Number of parallel simulated envs (default: 9).
      --no-finetune    Use exp_name=CLoSD_no_finetune instead of CLoSD_t2m_finetune.
      --headless       Run without the viewer window (headless=True).
      --dry-run        Print the python command that would run, but don't execute it.
  -h, --help           Show this message.

Examples:
  ./scripts/run_prompt_sequence.sh "a person squats down" "a person claps"
  ./scripts/run_prompt_sequence.sh "a person walks forward" "a person jumps" -s 4
  ./scripts/run_prompt_sequence.sh "a person waves" "a person sits down" --headless --dry-run

Notes:
  * Requires the 'closd' conda env to be active (conda activate closd) and a
    working GPU driver (nvidia-smi).
  * Hydra's override parser can be picky about prompts containing '=' or ','.
    Simple action phrases work best. Prompts must not contain the '|' character
    (it is used internally as the sequence separator).
EOF
}

# ---- defaults ----------------------------------------------------------------
num_envs=9
seconds_per_prompt=3
exp_name="CLoSD_t2m_finetune"
headless=false
dry_run=false

# ---- argument parsing --------------------------------------------------------
# Any non-option argument is a prompt (in order); the rest are options.
prompts=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -s|--seconds-per-prompt)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a value." >&2
                exit 1
            fi
            seconds_per_prompt="$2"
            shift 2
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
            while [ $# -gt 0 ]; do prompts+=("$1"); shift; done
            ;;
        -*)
            echo "Error: unknown option '$1'." >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
        *)
            prompts+=("$1")
            shift
            ;;
    esac
done

# ---- sanity checks -----------------------------------------------------------
if [ "${#prompts[@]}" -lt 2 ]; then
    echo "Error: provide at least two prompts (got ${#prompts[@]})." >&2
    echo "For a single prompt, use ./scripts/run_prompt.sh instead." >&2
    echo >&2
    usage >&2
    exit 1
fi

for p in "${prompts[@]}"; do
    if [[ "$p" == *"|"* ]]; then
        echo "Error: prompts must not contain the '|' character: '$p'." >&2
        echo "'|' is used internally to separate the sequence." >&2
        exit 1
    fi
done

# Validate num_envs is a positive integer.
if ! [[ "$num_envs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-envs must be a positive integer (got '$num_envs')." >&2
    exit 1
fi

# Validate seconds_per_prompt is a positive integer.
if ! [[ "$seconds_per_prompt" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --seconds-per-prompt must be a positive integer (got '$seconds_per_prompt')." >&2
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

# ---- build sequence strings --------------------------------------------------
# frames per prompt = seconds_per_prompt * 30 (sim runs the viewer at 30fps).
frames_per_prompt=$((seconds_per_prompt * 30))

prompt_seq=""
frames_seq=""
total_frames=0
for p in "${prompts[@]}"; do
    if [ -z "$prompt_seq" ]; then
        prompt_seq="$p"
        frames_seq="$frames_per_prompt"
    else
        prompt_seq="$prompt_seq|$p"
        frames_seq="$frames_seq|$frames_per_prompt"
    fi
    total_frames=$((total_frames + frames_per_prompt))
done

# Give the episode enough length to play the whole sequence, plus a margin so the
# last prompt isn't truncated. Only raise above the config default (300) if needed.
episode_length=$((total_frames + 60))
if [ "$episode_length" -lt 300 ]; then
    episode_length=300
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
    "env.dip.custom_prompt_sequence=$prompt_seq"
    "env.dip.custom_prompt_sequence_frames=$frames_seq"
    "env.episode_length=$episode_length"
)
if [ "$headless" = true ]; then
    cmd+=(headless=True)
else
    cmd+=(headless=False no_virtual_display=True)
fi

# ---- dry-run or execute ------------------------------------------------------
if [ "$dry_run" = true ]; then
    echo "Would run (from $repo_root):"
    # Print each token quoted so the prompts are unambiguous.
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

echo "Running CLoSD with prompt sequence: \"$prompt_seq\""
echo "  (${#prompts[@]} prompts, ${seconds_per_prompt}s each = ${frames_per_prompt} frames, episode_length=${episode_length}, num_envs=${num_envs}, exp_name=${exp_name})"
exec "${cmd[@]}"
