#!/bin/bash
#SBATCH --job-name=lr_wm1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=secondary
#SBATCH --account=bbradlyn-ic
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-10
#SBATCH --begin=now
module load anaconda/2023-Mar/3
module load cuda/11.7
# nvcc --version
# nvidia-smi

sleep $(($SLURM_ARRAY_TASK_ID * 5))

# Activate the Conda environment
#--account=bbradlyn-phys-eng
source activate torch_env

config=config_average.txt

learning_rate_input=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $12}' $config)

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
data_seed_start=0
data_seed_end=3
weight_decay=0
epochs=10000
weight_multiplier=10


#Run that file hombre - note the one for the cluster argument so I don't have to keep changing between local and cluster!
srun python3 cluster_run_average.py --data_seed_start=${data_seed_start} --data_seed_end=${data_seed_end} --weight_multiplier=${weight_multiplier} --weight_decay=${weight_decay} --epochs=${epochs} --learning_rate_input=${learning_rate_input} --cluster_arg=1

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
echo "This is array task ${SLURM_ARRAY_TASK_ID}, ${data_seed_start} ${data_seed_end} ${sgd_seed_start} ${sgd_seed_end} ${init_seed_start} ${init_seed_end} ${wd} ${grok} ${train_size} ${hl_size} ${lr_input} ${train_type} ${P} ${train_fraction} ${weight_mutliplier}" >> output.txt

#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script


# Deactivate the environment
# conda deactivate
