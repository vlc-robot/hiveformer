#!/bin/bash
#SBATCH --job-name=eval_tst
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out

set -x
set -e

module purge
pwd; hostname; date

cd $WORK/codes/hiveformer

. $WORK/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=$WORK/miniconda3/envs/bin/lib:$LD_LIBRARY_PATH

conda activate hiveformer
export PYTHONPATH=$PYTHONPATH:$(pwd)


export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

dirname=$1
seed=$2
checkpoint_step=$3

outdir=data/exprs/transformer_unet/${dirname}/seed${seed}

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    singularity exec \
    --bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
    $SINGULARITY_ALLOWED_DIR/vlc_rlbench.sif \
    xvfb-run -a python eval_models.py \
    --exp_config  ${outdir}/logs/training_config.yaml \
    --seed 200 --num_demos 500 \
    --checkpoint ${outdir}/ckpts/model_step_${checkpoint_step}.pt