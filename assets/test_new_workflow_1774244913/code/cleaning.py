import pandas as pd
import numpy as np
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

# 读取原始数据
df = pd.read_csv('/Users/cjialin/code/AutoMLByLLM/train.csv')
original_shape = df.shape
print(f\