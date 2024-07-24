import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import datasets
import pandas as pd
import numpy as np
import matplotlib as plt
import soundata

##importations done

#prepare datasets

dataset = soundata.initialize("urbansound8k")
dataset.download() #download the dataset
dataset.validate() #validate the expected files are present

#test
#test 2


## define a basic model
