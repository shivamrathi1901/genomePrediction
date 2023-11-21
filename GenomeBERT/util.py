import pandas as pd
# import random, re
# import numpy as np
import subprocess, os, glob

def create_test_train_val_file(data_dir):
    if(not os.path.isdir("data")):
      os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/valid")
    os.makedirs("data/test")
    for file in glob.glob(f"{data_dir}/*.csv"):
      rawdata = pd.read_csv(file)
      rawdata = rawdata[['Sequence', 'OC']]
      rawdata = rawdata.dropna()
      train_set, val_set, test_set = split_dataset(rawdata, split_ratio=0.3)
      train_set.to_csv(f"data/train/{file.split('/')[-1]}")
      val_set.to_csv(f"data/valid/{file.split('/')[-1]}")
      test_set.to_csv(f"data/test/{file.split('/')[-1]}")
    print("Split dataset complete!")


def dataload(file):
    rawdata = pd.read_csv(file)
    rawdata = rawdata[['Sequence', 'OC']]
    rawdata = rawdata.dropna()
    return rawdata

def split_dataset(rawdata, split_ratio=0.2):
    from sklearn.model_selection import train_test_split
    train_set, test_val_set = train_test_split(rawdata, test_size=split_ratio, random_state=42)
    val_set, test_set = train_test_split(test_val_set, test_size=0.5, random_state=42)
    return train_set, val_set, test_set

def plot(training_loss, val_loss, job_id):
    import matplotlib.pyplot as plt
    plt.plot(range(1, epochs + 1), training_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(f"{job_id}_plot.png")

def count_nucleotides(sequence,nucleotides_count):
  for nucleotide in sequence:
    if nucleotide in nucleotides_count:
      nucleotides_count[nucleotide] += 1
    else:
      nucleotides_count[nucleotide] = 1
  return nucleotides_count

def get_gpu_utilization(logger):
    try:
        output = subprocess.check_output(['nvidia-smi', '--format=csv,noheader,nounits', '--query-gpu=utilization.gpu'])
        gpu_utilization = [float(x) for x in output.decode('utf-8').strip().split('\n')]
        logger.info("GPU Utilization: {}%".format(gpu_utilization))
        return
    except Exception as e:
        logger.warning(f"Failed to get GPU utilization: {e}")
        return None