#!/bin/bash
#SBATCH --partition=highmem_p               # Partition (job queue)
#SBATCH --job-name=test               # Assign an short name to your job
##SBATCH --gres=gpu:A100:1               # Assign a GPU type:count
##SBATCH --reservation=ah_res            # Attach job to specific nodetype
##SBATCH --nodelist=b6-1                 # nodename
#SBATCH --nodes=1                       # Number of nodes you require
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=16              # Cores per task (>1 if multithread tasks)
#SBATCH --mem=256G                      # Real memory (RAM) required (MB)
#SBATCH --time=7:00:00               # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                    # Export you current env to the job env
#SBATCH --mail-user=soumya.bharadwaj@uga.edu  # email of the user
#SBATCH --mail-type=END,FAIL            # Notify user by email when certain event types occur
#SBATCH --output=log/%j_%x.out  # STDOUT output file

##Install of softwares and load before running models:
ml Python/3.8.6-GCCcore-10.2.0
# pip install numpy==1.22.3
# pip install transformers==4.35.2
export TOKENIZERS_PARALLELISM=false
# pip install pycuda 
# pip install scikit-learn scipy matplotlib pycuda lietorch
# pip uninstall triton -y
date
python test.py models/GenomeBERT $SLURM_JOB_ID 
date

