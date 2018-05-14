import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
##################################################
#                   LOAD DATA                    #
##################################################
bank_data = pd.read_csv('C:/Users/zinaw/Google Drive/Q4_2017/Q4_Cyber/cyber_lab1/data/after_one_hot.csv')# print(bank_data.astype(bool).sum(axis=0))#345 1%
bank_data=bank_data[~bank_data['mail_id'].isnull()]
bank_data = bank_data.drop('4', 1)
print(bank_data.iloc[1])
bank_data = (bank_data - bank_data.min()) / (bank_data.max() - bank_data.min())
print(bank_data.iloc[1])
bank_data.to_csv('after_normalization.csv')
bank_data = pd.read_csv('C:/Users/zinaw/git/Q4_Cyber/ClassifierPart/after_normalization.csv')# print(bank_data.astype(bool).sum(axis=0))#345 1%
print(bank_data.isnull().values.any())
# print(bank_data.iloc[2])