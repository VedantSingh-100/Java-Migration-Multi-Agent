#!/bin/bash
#SBATCH --job-name=migration_sonnet35
#SBATCH --output=slurm_logs/sonnet35_%j.out
#SBATCH --error=slurm_logs/sonnet35_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0

# =============================================================================
# Java Migration - Claude 3.5 Sonnet
# =============================================================================

echo "=============================================="
echo "Java Migration - Claude 3.5 Sonnet"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=============================================="

cd /home/vhsingh/Java_Migration

# Activate conda environment
source /home/vhsingh/miniconda3/etc/profile.d/conda.sh
conda activate java_migration

# Set Maven home
export MAVEN_HOME=/home/vhsingh/apache-maven-3.9.11
export PATH=$MAVEN_HOME/bin:$PATH

# Set model ID
export BEDROCK_MODEL_ID="us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# Set model-specific output directories
export REPOS_DIR=/home/vhsingh/Java_Migration/repositories_sonnet35
export LOGS_DIR=/home/vhsingh/Java_Migration/logs_sonnet35

# Create directories
mkdir -p $REPOS_DIR $LOGS_DIR

echo "Model: Claude 3.5 Sonnet"
echo "Repos: $REPOS_DIR"
echo "Logs: $LOGS_DIR"
echo "=============================================="

# Run migration with model-specific CSV
python migrate_all_repos.py --csv docs/selected40_sonnet35.csv

echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="
