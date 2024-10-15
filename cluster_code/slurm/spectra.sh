#!/bin/bash
#SBATCH --job-name=sp
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --partition=IllinoisComputes
#SBATCH --account=bbradlyn-ic
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-1
#SBATCH --mem=16G
#SBATCH --begin=now
module load anaconda/2023-Mar/3
module load cuda/11.7
# nvcc --version
# nvidia-smi

sleep $(($SLURM_ARRAY_TASK_ID * 5))

# Activate the Conda environment
#--account=bbradlyn-phys-eng
source activate torch_env

config=config_spectra.txt

thetadeg=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
Uorb=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
Urot=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

# Other terms you want to set in the yaml
particles=8
# data_seed_start=0
# data_seed_end=3
# weight_decay=0
# epochs=10000
# weight_multiplier=10


#Run that file hombre - note the one for the cluster argument so I don't have to keep changing between local and cluster!
srun python3 ../main_spectrum.py --thetadeg=${thetadeg} --Uorb=${Uorb} --Urot=${Urot}  --particles=${particles}

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
echo "This is array task ${SLURM_ARRAY_TASK_ID}, dir ${sub_dir}, term ${term_number}, particles ${particles}" >> output.txt

#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script


# Deactivate the environment
# conda deactivate
