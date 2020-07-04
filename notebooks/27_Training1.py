


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
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)




from tqdm import tqdm
tqdm.pandas()




def seed_everything(seed=51):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




seed_everything(seed=51)




sample_submission = pd.read_csv('../input/sample_submission.csv')




sample_submission['is_evaluation'] = sample_submission['id'].apply(lambda x: 1 if x.split('_')[-1] == 'evaluation' else 0)




validation = sample_submission[sample_submission['is_evaluation'] == 0].reset_index(drop=True)
evaluation = sample_submission[sample_submission['is_evaluation'] == 1].reset_index(drop=True)




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




train = pd.read_pickle('train1_1.pkl')
val = pd.read_pickle('val1_1.pkl')
print(train.shape,val.shape)




del train['sold'],val['sold']




train = reduce_mem_usage(train)
val = reduce_mem_usage(val)




train.head()




print(train.shape)
train = pd.concat([train,val],axis=0).reset_index(drop=True)
del val
print(train.shape)
gc.collect()




cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id']
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "dayofyear","year","turnover","day_number","month"]




train.head()








train_cols = train.columns[~train.columns.isin(useless_cols)]
























print(list(train_cols))




print(len(train_cols))




X_train = train[train_cols]
y_train = train["sales"]




train[train_cols].head()




train_data = lgb.Dataset(X_train , label = y_train,categorical_feature=cat_feats, free_raw_data=False)








params = {
"objective" : "poisson",
"metric" :"rmse",
"force_row_wise" : False,
"learning_rate" : 0.02,
 "feature_fraction" : 1.0,
"sub_row" : 0.75,
"bagging_freq" : 1,
"lambda_l2" : 0.1,
'verbosity': 1,
'num_iterations' : 1274,
'num_leaves': 64,
"min_data_in_leaf": 100,
}




m_lgb = lgb.train(params, train_data, verbose_eval= 50) 




m_lgb.save_model("model_1.lgb")




pd.Series(m_lgb.feature_importance('gain'),index=m_lgb.feature_name()).sort_values()








"""
Training until validation scores don't improve for 25 rounds.
[50]	valid_0's rmse: 3.05769
[100]	valid_0's rmse: 2.69825
[150]	valid_0's rmse: 2.54086
[200]	valid_0's rmse: 2.47269
[250]	valid_0's rmse: 2.44135
[300]	valid_0's rmse: 2.42463
[350]	valid_0's rmse: 2.41536
[400]	valid_0's rmse: 2.40923
[450]	valid_0's rmse: 2.40532
[500]	valid_0's rmse: 2.40226
[550]	valid_0's rmse: 2.39882
[600]	valid_0's rmse: 2.39598
[650]	valid_0's rmse: 2.39383
[700]	valid_0's rmse: 2.3911
[750]	valid_0's rmse: 2.38946
[800]	valid_0's rmse: 2.38828
[850]	valid_0's rmse: 2.38754
[900]	valid_0's rmse: 2.38633
[950]	valid_0's rmse: 2.38612
[1000]	valid_0's rmse: 2.38547
[1050]	valid_0's rmse: 2.38431
[1100]	valid_0's rmse: 2.38393
[1150]	valid_0's rmse: 2.38373
[1200]	valid_0's rmse: 2.3829
[1250]	valid_0's rmse: 2.38252
Early stopping, best iteration is:
[1274]	valid_0's rmse: 2.38225
"""




"""
Training until validation scores don't improve for 25 rounds.
[50]	valid_0's rmse: 3.07038
[100]	valid_0's rmse: 2.7132
[150]	valid_0's rmse: 2.55708
[200]	valid_0's rmse: 2.48791
[250]	valid_0's rmse: 2.45609
[300]	valid_0's rmse: 2.43962
[350]	valid_0's rmse: 2.42953
[400]	valid_0's rmse: 2.42298
[450]	valid_0's rmse: 2.41843
[500]	valid_0's rmse: 2.41465
[550]	valid_0's rmse: 2.41161
[600]	valid_0's rmse: 2.40842
[650]	valid_0's rmse: 2.40572
[700]	valid_0's rmse: 2.40291
[750]	valid_0's rmse: 2.40021
[800]	valid_0's rmse: 2.39833
[850]	valid_0's rmse: 2.39731
[900]	valid_0's rmse: 2.39626
[950]	valid_0's rmse: 2.39547
[1000]	valid_0's rmse: 2.39463
[1050]	valid_0's rmse: 2.39432
[1100]	valid_0's rmse: 2.39379
Early stopping, best iteration is:
[1106]	valid_0's rmse: 2.39362
"""




base_score = 2.38225




from sklearn.metrics import mean_squared_error
from math import sqrt




import eli5
from eli5.sklearn import PermutationImportance




used_list = []
dropped_list = []
for i in tqdm(train_cols):
    X_val2 = X_val.copy()
    X_val2[i] = np.random.permutation(X_val2[i])
    predicted = m_lgb.predict(X_val2)
    feature_score = sqrt(mean_squared_error(y_val,predicted))
    print(i)
    print(base_score - feature_score)
    print("*"*50)
    if base_score - feature_score >= 0:
        dropped_list.append(i)
    else:
        used_list.append(i)




print(len(used_list))
print(len(dropped_list))




print(used_list)




print(dropped_list)














plt.figure(figsize=(20,10))
pd.Series(m_lgb.feature_importance(),index=train_cols).sort_values()#.plot(kind='barh')

