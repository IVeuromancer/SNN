import tonic
import tonic.transforms as transforms  # Not to be mistaken with torchdata.transfroms
from tonic import DiskCachedDataset
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch.spikeplot as splt
from snntorch import functional as SF
from snntorch import utils
import matplotlib.pyplot as plt
# from IPython.display import HTML
# from IPython.display import display
import numpy as np
import torchdata
import os
# from ipywidgets import IntProgress
import time
import statistics

root = "/home/jane_simko/SNN/STMNIST_dataset"
os.listdir(root)
dataset = tonic.prototype.datasets.STMNIST(root=root,
                                           keep_compressed = False, shuffle = False)