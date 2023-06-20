#!/bin/bash
#SBATCH --job-name=trainbc
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

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

taskvars="pick_and_lift+0,pick_up_cup+0,put_knife_on_chopping_board+0,put_money_in_safe+0,push_button+0,reach_target+0,slide_block_to_target+0,stack_wine+0,take_money_out_safe+0,take_umbrella_out_of_umbrella_stand+0"
seed=$1

configfile=config/transformer_unet.yaml

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    singularity exec \
    --bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE --nv \
    $SINGULARITY_ALLOWED_DIR/vlc_rlbench.sif \
    xvfb-run -a python train_models.py \
    --exp-config ${configfile} \
    output_dir data/exprs/transformer_unet/10tasks \
    DATASET.taskvars ${taskvars} \
	DATASET.data_dir data/train_dataset/keysteps/seed${seed}
	