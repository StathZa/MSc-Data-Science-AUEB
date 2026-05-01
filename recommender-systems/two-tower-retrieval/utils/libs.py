# Basic libraries
import os, glob, sys, re, random, math, gc, pickle, logging
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass

# Paths, Data Types and Strings manipulation
from pathlib import Path
import inspect
from typing import Literal, List, Union, Any, get_args

# Mount Drive
from utils.config import USE_COLAB 
print(USE_COLAB)
if 'google.colab' in sys.modules and USE_COLAB:
  from google.colab import drive
  drive.mount('/content/drive', force_remount=True)

# Filtering warnings
if not sys.warnoptions: 
  import warnings
  warnings.filterwarnings('ignore')

# Deep Learning Pytorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer