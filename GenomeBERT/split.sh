#!/bin/bash

#SBATCH --partition=batch               # Partition (job queue)
#SBATCH --job-name=split                # Assign an short name to your job
##SBATCH --gres=gpu:A100:4              # Assign a GPU type:count
##SBATCH --reservation=ah_res           # Attach job to specific nodetype
##SBATCH --nodelist=b6-2                        # nodename
#SBATCH --nodes=1                       # Number of nodes you require
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1               # Cores per task (>1 if multithread tasks)
#SBATCH --mem=200G                      # Real memory (RAM) required (MB)
#SBATCH --time=7-00:00:00               # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                    # Export you current env to the job env
#SBATCH --mail-type=END,FAIL            # Notify user by email when certain event types occur
#SBATCH --mail-user=ratish.jha@uga.edu  # email of the user
#SBATCH --output=log/%j_%x.out  # STDOUT output file

##Install of softwares and load before running models:
ml Python/3.8.6-GCCcore-10.2.0

# pip install transformers==4.30.2

python dataprep.py /work/ah2lab/aisdata
