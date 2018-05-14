import pickle
import pandas as pd 
from random import randint

# only_categorical_data = pd.DataFrame()
# one_hot_file = []
# one_hot_file = pd.read_pickle('after_one_hot1')
# print(len(one_hot_file))
# temp = []
# only_categorical_data_dict = {}
# for index, element in enumerate(one_hot_file):
# 	temp.append({k: v for k, v in enumerate(element[0])})
# 	print(len(temp))
# after_one_hot_df = pd.DataFrame(temp)
# print(after_one_hot_df.shape)
# after_one_hot_df.to_csv('only_categorical_data.csv', encoding='utf-8', index=False)

only_numerical_data = pd.read_csv('only_numerical_data.csv')
only_categorical_data = pd.read_csv('only_categorical_data.csv')
after_one_hot = pd.concat([only_numerical_data, only_categorical_data], axis=1)
print(after_one_hot.iloc[0,:])
print(only_numerical_data.shape, only_categorical_data.shape, after_one_hot.shape)
after_one_hot.to_csv('after_one_hot.csv', encoding='utf-8', index=False)

