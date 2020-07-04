


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from lightgbm import LGBMClassifier,LGBMRegressor
import lightgbm as lgb
import tensorflow as tf
import dask.dataframe as dd
import math
import random
import os
import gc
import sys
plt.style.use('fivethirtyeight')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)




['item_id', 'dept_id', 'store_id', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'sales_lag_28', 'sales_lag_28_roll_mean_7', 'sales_lag_28_roll_std_7', 'sales_lag_28_roll_max_7', 'sales_lag_28_roll_mean_28', 'sales_lag_28_roll_max_28', 'sales_lag_91', 'sales_lag_91_roll_mean_7', 'sales_lag_91_roll_std_7', 'sales_lag_91_roll_mean_28', 'turnover_lag_28', 'turnover_lag_28_roll_mean_7', 'turnover_lag_91', 'turnover_lag_91_roll_mean_7', 'turnover_lag_91_roll_std_7', 'turnover_lag_91_roll_max_7', 'is_sport', 'is_holiday', 'sales_lag_28_roll_mean_91', 'sales_lag_28_roll_std_91', 'sales_lag_28_roll_max_91', 'sales_lag_28_roll_mean_182', 'sales_lag_28_roll_std_182', 'sales_lag_91_roll_mean_91', 'sales_lag_91_roll_max_91', 'sales_lag_91_roll_mean_182', 'sales_lag_182', 'sales_lag_182_roll_mean_182', 'sales_lag_182_roll_std_182', 'sales_lag_182_roll_max_182', 'sales_lag_364', 'sales_lag_364_roll_mean_7', 'sales_lag_364_roll_std_7', 'sales_lag_364_roll_max_7', 'sales_lag_364_roll_mean_28', 'sales_lag_364_roll_std_28', 'sales_lag_364_roll_max_28']




['turnover_lag_364','turnover_lag_182','turnover_lag_364_roll_std_91',
 'turnover_lag_182_roll_mean_182','turnover_lag_91_roll_mean_182',
'turnover_lag_182_roll_std_182','turnover_lag_28_roll_mean_91']




used_index = [9747090,12689405,17844519,40666139,52063828]




from tqdm import tqdm
tqdm.pandas()




def seed_everything(seed=51):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




seed_everything(seed=51)




get_ipython().run_cell_magic('time', '', "data = pd.read_pickle('data.pkl')")




data['sold'] = data['sell_price'].notnull() * 1




data.tail()




for i in tqdm([28,56,91,182,364,728]):
    data['sales_lag_{}'.format(i)] = data.groupby('id')['sales'].shift(i)




get_ipython().run_cell_magic('time', '', "data['turnover'] = data['sales'] * data['sell_price']\ndata['turnover'] = data['turnover'].fillna(0.0)")




for i in tqdm([28,56,91,182,364,728]):
    data['turnover_lag_{}'.format(i)] = data.groupby('id')['turnover'].shift(i)




i,j = 28,7
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['sales_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
data['sales_lag_{}_roll_max_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).max())
print("*"*50)
i,j = 28,28
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['sales_lag_{}_roll_max_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).max())
print("*"*50)
i,j = 28,91
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['sales_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
data['sales_lag_{}_roll_max_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).max())
print("*"*50)
i,j = 28,182
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['sales_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
print("*"*50)
i,j = 91,7
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['sales_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
print("*"*50)
i,j = 91,28
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
print("*"*50)
i,j = 91,91
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
print("*"*50)
i,j = 91,182
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
print("*"*50)
i,j = 182,182
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['sales_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
print("*"*50)
i,j = 364,7
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['sales_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
data['sales_lag_{}_roll_max_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).max())
print("*"*50)
i,j = 364,28
data['sales_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['sales_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
data['sales_lag_{}_roll_max_{}'.format(i,j)]= data.groupby('id')['sales_lag_{}'.format(i)].transform(lambda x : x.rolling(j).max())
print("*"*50)




i,j = 28,7
data['turnover_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
print("*"*50)
i,j = 28,91
data['turnover_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
print("*"*50)
i,j = 91,7
data['turnover_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['turnover_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
data['turnover_lag_{}_roll_max_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).max())
print("*"*50)
i,j = 91,182
data['turnover_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
print("*"*50)
i,j = 182,182
data['turnover_lag_{}_roll_mean_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).mean())
data['turnover_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
print("*"*50)
i,j = 364,91
data['turnover_lag_{}_roll_std_{}'.format(i,j)]= data.groupby('id')['turnover_lag_{}'.format(i)].transform(lambda x : x.rolling(j).std())
print("*"*50)




def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




data = reduce_mem_usage(data)




data.head()




from sklearn.preprocessing import LabelEncoder




for i in tqdm(['item_id','dept_id','cat_id','store_id','state_id']):
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i]) 




print(data.shape)
data = data[(data['sold'] == 1.0)]
print(data.shape)




data['day_number'] = data['d'].progress_apply(lambda x: int(x.split('_')[1]))




print(data.shape)
data = data[(data['day_number'] >= 1069)]
print(data.shape)




other_list = ["Father's day","Mother's day","ValentinesDay","NewYear"]
cl1 = 'event_name_1'
cl2 = 'event_name_2'
data['is_other'] = (data[cl1].isin(other_list) * 1 + data[cl2].isin(other_list) * 1 > 0) *1
data['is_holiday'] = (data[cl1].notnull() * 1 + data[cl2].notnull() * 1 > 0) *1




data['dayofyear'] = pd.to_datetime(data['date']).dt.dayofyear
data['year'] = pd.to_datetime(data['date']).dt.year




data['month'] = data['date'].progress_apply(lambda x: x.split('-')[1])




data['month'] = data['month'].progress_apply(int)




print(data.shape)
data = data[ data['month'] < 7 ]
print(data.shape)




del data['event_name_1'],data['event_type_1'],data['event_name_2'],data['event_type_2']
gc.collect()




data.head()




data[(data['year'] == 2014) & (data['dayofyear'] < 143) ].to_pickle('train1_1.pkl')
data[(data['year'] == 2014) & (data['dayofyear'] >= 143) & (data['dayofyear'] < 171)].to_pickle('val1_1.pkl')
print("*"*50)
data[(data['year'] == 2015) & (data['dayofyear'] < 143) ].to_pickle('train2_1.pkl')
data[(data['year'] == 2015) & (data['dayofyear'] >= 143) & (data['dayofyear'] < 171)].to_pickle('val2_1.pkl')
print("*"*50)
data[(data['year'] == 2016) & (data['dayofyear'] < 144-56) ].to_pickle('train3_1.pkl')
data[(data['year'] == 2016) & (data['dayofyear'] >= 144-56) & (data['dayofyear'] < 144-28)].to_pickle('val3_1.pkl')
print("*"*50)
data[(data['year'] == 2016) & (data['dayofyear'] >= 144-28) & (data['dayofyear'] < 144)].to_pickle('testpart1_1.pkl')
data[(data['year'] == 2016) & (data['dayofyear'] >= 144) & (data['dayofyear'] < 172)].to_pickle('testpart2_1.pkl')





