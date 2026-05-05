# Basic libraries
import os, gc, glob, sys, re, random, math, gc, pickle, logging
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass

# Environment files
from dotenv import load_dotenv, dotenv_values

# progress bar
from tqdm import tqdm

# Paths, Data Types and Strings manipulation
from pathlib import Path
import inspect
from typing import Literal, List, Union, Any, get_args

# Mount Drive
from utils.config import USE_COLAB 

if 'google.colab' in sys.modules and USE_COLAB:
  from google.colab import drive
  drive.mount('/content/drive', force_remount=True)

from google.colab import userdata

# Filtering warnings
if not sys.warnoptions: 
  import warnings
  warnings.filterwarnings('ignore')

# Data loading
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Text Embeddings
from sentence_transformers import SentenceTransformer

# Deep Learning Pytorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer