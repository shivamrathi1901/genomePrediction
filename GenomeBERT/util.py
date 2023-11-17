import pandas as pd
import random, re
import numpy as np


def dataload(file):
    rawdata = pd.read_csv(file)
    rawdata = rawdata[['Sequence', 'OC']]
    return split_dataset(rawdata)

def split_dataset(rawdata):
    from sklearn.model_selection import train_test_split
    train_set, test_val_set = train_test_split(rawdata, test_size=0.3, random_state=42)
    val_set, test_set = train_test_split(test_val_set, test_size=0.5, random_state=42)
    return train_set, val_set, test_set


def count_nucleotides(sequence,nucleotides_count):
  for nucleotide in sequence:
    if nucleotide in nucleotides_count:
      nucleotides_count[nucleotide] += 1
    else:
      nucleotides_count[nucleotide] = 1
  return nucleotides_count
