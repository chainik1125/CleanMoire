#!/bin/bash

# Define variables
LOCAL_SCRIPT="/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/Code/slurm/run_script_config_seedavg.sh"
LOCAL_CODE_DIR="/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/CleanMoire/Code"
REMOTE_CODE_DIR="/projects/illinois/eng/physics/bbradlyn/dmitry2/Moire"
REMOTE_USER="dmitry2"
REMOTE_SERVER="cc-login.campuscluster.illinois.edu"
REMOTE_DIR="/projects/illinois/eng/physics/bbradlyn/dmitry2/Moire/Code/slurm"


# Copy the code folder to the SLURM server
scp -r "$LOCAL_CODE_DIR" "$REMOTE_USER@$REMOTE_SERVER:$REMOTE_CODE_DIR"

# Copy the script to the SLURM server
#scp $LOCAL_SCRIPT $REMOTE_USER@$REMOTE_SERVER:$REMOTE_DIR

# Execute the script on the SLURM server
# ssh "$REMOTE_USER@$REMOTE_SERVER" << EOF
# cd "$REMOTE_DIR"
# sbatch $(basename "$LOCAL_SCRIPT")
# EOF