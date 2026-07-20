# Sourced by every launcher and by every sbatch wrapper.
# Loads modules, activates .venv, points caches at $SCRATCH, and defines submit().
#
# Adding a cluster: add a hostname branch to the SCRATCH block and a matching
# branch to the submit() dispatch at the bottom, plus setup/submit_<cluster>.sbatch.

_host=$(hostname)
_host_f=$(hostname -f 2>/dev/null || echo "$_host")

# fir is an Alliance cluster: it needs the software stack loaded before cuda.
if [[ "$_host_f" == *.fir.alliancecan.ca ]]; then
    module load StdEnv/2023
fi
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

source .venv/bin/activate

# Killarney (Vector): scratch lives under the user's home, not /scratch.
if [[ "$_host" == klogin* || "$_host_f" == *.paice.vectorinstitute.ai ]]; then
    export SCRATCH=/home/$USER/scratch/$USER
else
    export SCRATCH=/scratch/$USER
fi

export PROTEUS_OUTPUT_DIR=${PROTEUS_OUTPUT_DIR:-$SCRATCH/proteus}
export HF_HOME=$SCRATCH/huggingface
export HF_DATASETS_CACHE=$SCRATCH/huggingface/datasets
export TORCH_HOME=$SCRATCH/torch

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Compute nodes without internet must read every model from the HF cache.
if [[ $OFFLINE_MODE == 1 ]]; then
    echo "Running in offline mode"
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
fi

# True if the job is already queued/running, or completed in the past 2 days.
# Makes every launcher idempotent: re-running only submits what is missing.
function should_skip_job() {
    local job_name="$1"

    if squeue --name="$job_name" --user="$USER" --noheader 2>/dev/null | grep -q .; then
        echo "SKIP: Job '$job_name' is already running or pending."
        return 0
    fi

    local two_days_ago
    two_days_ago=$(date -d '2 days ago' +%Y-%m-%d 2>/dev/null || date -v-2d +%Y-%m-%d)
    if sacct --name="$job_name" --user="$USER" --starttime="$two_days_ago" \
        --state=COMPLETED --noheader 2>/dev/null | grep -q .; then
        echo "SKIP: Job '$job_name' completed successfully in the past 2 days."
        return 0
    fi

    return 1
}

if [[ "$_host" == klogin* || "$_host_f" == *.paice.vectorinstitute.ai ]]; then
    function submit() {
        local job_name="$1"
        local command="$2"
        should_skip_job "$job_name" && return 0
        mkdir -p logs
        sbatch --job-name="$job_name" \
               --output="logs/%j_$job_name.out" \
               --error="logs/%j_$job_name.out" \
               setup/submit_killarney.sbatch "$command"
    }
elif [[ "$_host_f" == *.fir.alliancecan.ca ]]; then
    function submit() {
        local job_name="$1"
        local command="$2"
        should_skip_job "$job_name" && return 0
        mkdir -p logs
        sbatch --job-name="$job_name" \
               --output="logs/%j_$job_name.out" \
               --error="logs/%j_$job_name.out" \
               setup/submit_fir.sbatch "$command"
    }
else
    echo "Unknown hostname: $_host - cannot define submit(). See setup/start_env.sh."
fi
