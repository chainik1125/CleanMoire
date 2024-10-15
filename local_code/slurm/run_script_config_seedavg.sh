#!/bin/bash
#SBATCH --job-name=ni_tensor
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --partition=IllinoisComputes
#SBATCH --account=bbradlyn-ic
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-1
#SBATCH --begin=now
module load anaconda/2023-Mar/3
module load cuda/11.7
# nvcc --version
# nvidia-smi

sleep $(($SLURM_ARRAY_TASK_ID * 5))

# Activate the Conda environment
#--account=bbradlyn-phys-eng
source activate torch_env

config=config_make_terms.txt

sub_dir=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
term_number=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

# Other terms you want to set in the yaml
particles=8
# data_seed_start=0
# data_seed_end=3
# weight_decay=0
# epochs=10000
# weight_multiplier=10


#Run that file hombre - note the one for the cluster argument so I don't have to keep changing between local and cluster!
srun python3 ../make_templates_tensor_cluster.py --sub_dir=${sub_dir} --term_number=${term_number} --particles=${particles}

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
echo "This is array task ${SLURM_ARRAY_TASK_ID}, dir ${sub_dir}, term ${term_number}, particles ${particles}" >> output.txt

#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script


# Deactivate the environment
# conda deactivate
