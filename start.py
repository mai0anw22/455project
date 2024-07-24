import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import datasets
import pandas as pd
import numpy as np
import soundata

##importations done

#prepare datasets

dataset = soundata.initialize("urbansound8k")
dataset.download() #download the dataset
dataset.validate() #validate the expected files are present

#test


## define a basic model
