# Hiveformer: History-aware instruction-conditioned multi-view transformer for robotic manipulation

This is a PyTorch re-implementation of the Hiveformer paper:
> Instruction-driven history-aware policies for robotic manipulations  
> Pierre-Louis Guhur, Shizhe Chen, Ricardo Garcia, Makarand Tapaswi, Ivan Laptev, Cordelia Schmid  
> **CoRL 2022 (oral)**


## Prerequisite

1. Installation. 

Option 1: Use our pre-build singularity image.
```
singularity pull library://rjgpinel/rlbench/vlc_rlbench.sif
```

Option 2: Install everything from scratch.
```bash
conda create --name hiveformer python=3.9
conda activate hiveformer
```

See instructions in [PyRep](https://github.com/stepjam/PyRep) and [RLBench](https://github.com/stepjam/RLBench) to install RLBench simulator (with VirtualGL in headless machines). Use our modified version of [RLBench](https://github.com/rjgpinel/RLBench) to support additional tasks.

```bash
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)
```


2. Dataset generation

Option 1: Use our [generated datasets](https://www.dropbox.com/s/zrth410b4voi4ut/train_dataset.tar.gz?dl=0) including the keystep trajectories and instruction embeddings.

Option 2: generate the dataset on your own.
```bash
seed=0
task=put_knife_on_chopping_board
variation=0
variation_count=1

# 1. generate microstep demonstrations
python preprocess/generate_dataset_microsteps.py \
     --save_path data/train_dataset/microsteps/seed{seed} \
    --all_task_file assets/all_tasks.json \
    --image_size 128,128 --renderer opengl \
    --episodes_per_task 100 \
    --tasks ${task} --variations ${variation_count} --offset ${variation} \
    --processes 1 --seed ${seed} 

# 2. generate keystep demonstrations
python preprocess/generate_dataset_keysteps.py \
    --microstep_data_dir data/train_dataset/microsteps/seed${seed} \
    --keystep_data_dir data/train_dataset/keysteps/seed${seed} \
    --tasks ${task}

# 3. (optional) check the correctness of generated keysteps
python preprocess/evaluate_dataset_keysteps.py \
    --microstep_data_dir data/train_dataset/microsteps/seed${seed} \
    --keystep_data_dir data/train_dataset/keysteps/seed${seed} \
     --tasks ${task}

# 4. generate instructions embeddings for the tasks
python preprocess/generate_instructions.py \
    --encoder clip \
    --output_file data/train_dataset/taskvar_instrs/clip
```



## Train

Our codes support distributed training with multiple GPUs in SLURM clusters.

For slurm users, please use the following command to launch the training script.
```bash
sbatch job_scripts/train_multitask_bc.sh
```

For non-slurm users, please manually set the environment variables as follows.

```bash
export WORLD_SIZE=1
export MASTER_ADDR='localhost'
export MASTER_PORT=10000

export LOCAL_RANK=0 
export RANK=0
export CUDA_VISIBLE_DEVICES=0

python train_models.py --exp-config config/transformer_unet.yaml
```



## Evaluation

For slurm users, please use the following command to launch the evaluation script.
```bash
sbatch job_scripts/eval_tst_split.sh
```

For non-slurm users, run the following commands to evaluate the trained model.

```bash
# set outdir to the directory of your trained model
export DISPLAY=:0.0 # in headless machines

# validation: select the best epoch
for step in {5000..300000..5000}
do
python eval_models.py \
    --exp_config ${outdir}/logs/training_config.yaml \
    --seed 100 --num_demos 20 \
    checkpoint ${outdir}/ckpts/model_step_${step}.pt
done

# run the script to summarize the validation results
python summarize_val_results.py --result_file ${outdir}/preds/seed100/results.jsonl

# test: use a different seed from validation
step=300000
python eval_models.py \
    --exp_config ${outdir}/logs/training_config.yaml \
    --seed 200 --num_demos 500 \
    checkpoint ${outdir}/ckpts/model_step_${step}.pt

# run the script to summarize the testing results
python summarize_tst_results.py --result_file ${outdir}/preds/seed200/results.jsonl
```

We also provided trained models in [Dropbox](https://www.dropbox.com/s/o4na7namn1ujhng/transformer_unet%2Bgripper_attn_multi32_300k.tar.gz?dl=0) for the multi-task setting (10 tasks).
You could obtain results as follows which are similar to the results in the paper:

|        | pick_ and_lift | pick_up _cup | put_knife_on_ chopping_board | put_money _in_safe | push_ button | reach_ target | slide_block _to_target | stack _wine | take_money _out_safe | take_umbrella_out_ of_umbrella_stand |  Avg. |
|:------:|:--------------:|:------------:|:----------------------------:|:------------------:|:------------:|:-------------:|:----------------------:|:-----------:|:--------------------:|:------------------------------------:|:-----:|
| seed=0 |      89.00     |     76.80    |             72.80            |        93.00       |     69.60    |     100.00    |          74.20         |    87.20    |         73.20        |                 89.80                | 82.56 |
| seed=2 |      91.40     |     75.80    |             76.20            |        81.60       |     86.60    |     100.00    |          85.00         |    89.00    |         72.80        |                 79.60                | 83.80 |
| seed=4 |      91.60     |     83.60    |             72.80            |        83.00       |     88.40    |     100.00    |          57.80         |    83.20    |         69.60        |                 89.60                | 81.96 |
|  Avg.  |      90.67     |     78.73    |             73.93            |        85.87       |     81.53    |     100.00    |          72.33         |    86.47    |         71.87        |                 86.33                | 82.77 |



