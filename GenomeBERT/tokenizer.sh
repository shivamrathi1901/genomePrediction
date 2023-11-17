#!/bin/bash

#SBATCH --partition=highmem_p          	# Partition (job queue)
#SBATCH --job-name=tokenize          	# Assign an short name to your job
##SBATCH --gres=gpu:A100:4	     	# Assign a GPU type:count
##SBATCH --reservation=ah_res         	# Attach job to specific nodetype
##SBATCH --nodelist=b6-2			# nodename 
#SBATCH --nodes=1                    	# Number of nodes you require
#SBATCH --ntasks=1                 	# total number of tasks across all nodes
#SBATCH --cpus-per-task=16        	# Cores per task (>1 if multithread tasks)
#SBATCH --mem=512G                  	# Real memory (RAM) required (MB)
#SBATCH --time=7-00:00:00              	# Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 	# Export you current env to the job env
#SBATCH --mail-user=ratish.jha@uga.edu	# email of the user
#SBATCH --output=slurm.%x_%j.out 	# STDOUT output file

##Install of softwares and load before running models:
ml Python/3.8.6-GCCcore-10.2.0


python train_tokenizer.py -m GenomeBERT -d /scratch/sb66469/ProjectAIS/FinalData -id $SLURM_JOB_ID 
