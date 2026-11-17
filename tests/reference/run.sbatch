#!/bin/bash
# The reference suite as one SLURM job, from the repository's root:
#   sbatch -A <account> -p <partition> tests/reference/run.sbatch [pytest options]
# Its solves are spread over the job's cores; the table is printed at the end of the log
# and written to reference-<job id>.md.
#SBATCH --job-name=fips-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=reference-%j.log
set -euo pipefail
export FIPS_REFERENCE_WORKERS="$SLURM_CPUS_PER_TASK"
export FIPS_REFERENCE_REPORT="${FIPS_REFERENCE_REPORT:-reference-$SLURM_JOB_ID.md}"
uv run just test-reference "$@"
