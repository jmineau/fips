#!/bin/bash
# The reference files' runs as an array of SLURM jobs, from the repository's root:
#   n=$(uv run --with nutpie==0.16.11 python -m tests.reference.generate jobs | wc -l)
#   sbatch -A <account> -p <partition> --array=0-$((n - 1)) tests/reference/generate.sbatch <directory>
# then, when every run has ended:
#   uv run --with nutpie==0.16.11 python -m tests.reference.generate combine <directory>
# Each run is independent and starts again from nothing, so a preempted one can be resubmitted alone.
#SBATCH --job-name=fips-reference-runs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=reference-run-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1
uv run --with nutpie==0.16.11 python -m tests.reference.generate run "$SLURM_ARRAY_TASK_ID" "$1"
