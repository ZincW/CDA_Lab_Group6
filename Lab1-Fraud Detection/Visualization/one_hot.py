# <<<<<<< HEAD
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd
# import pickle
# from random import randint

# ##################################################
# #                  LOAD DATA                     #
# ##################################################
# only_numerical_data = pd.read_csv('only_numerical_data.csv')
# only_numerical_data.dropna(axis=0, how='any')
# before_one_hot = pd.read_csv('preprocessed_data_for_one_hot.csv')
# before_one_hot.dropna(axis=0, how='any')
# print(before_one_hot.shape, only_numerical_data.shape)
# ##################################################
# #                   FUNCTION                     #
# ##################################################
# enc = OneHotEncoder()
# temp=[]
# after_one_hot = []
# accountcode_number = len(set(before_one_hot['accountcode']))
# cardverificationcodesupplied_number = len(set(before_one_hot['cardverificationcodesupplied']))
# cvcresponsecode_number = len(set(before_one_hot['cvcresponsecode']))
# issuercountrycode_number = len(set(before_one_hot['issuercountrycode']))
# shoppercountrycode_number = len(set(before_one_hot['shoppercountrycode']))
# shopperinteraction_number = len(set(before_one_hot['shopperinteraction']))
# txvariantcode_number = len(set(before_one_hot['txvariantcode']))
# # check if the dimensions are right
# print(accountcode_number, cardverificationcodesupplied_number,cvcresponsecode_number,
#      issuercountrycode_number,shoppercountrycode_number,shopperinteraction_number, txvariantcode_number,
#      accountcode_number+cardverificationcodesupplied_number+cvcresponsecode_number+
#      issuercountrycode_number+shoppercountrycode_number+shopperinteraction_number+txvariantcode_number)
# ##################################################
# #                   ONE HOT                      #
# ##################################################
# for row in before_one_hot.iterrows():
#     index, data = row
#     if data['cvcresponsecode'] >= 3:
#         data['cvcresponsecode'] == 3
#     temp.append(data.tolist())
#     # print(index)
# enc.fit(temp)
# print ("enc.n_values_ is:",enc.n_values_)
# print ("enc.feature_indices_ is:",enc.feature_indices_)
# print(len(temp))

# for row in before_one_hot.iterrows():
#     index, data = row
#     # print(index)
#     my_data = (np.asarray(data)).reshape(1, -1)
#     after_one_hot.append(enc.transform(my_data).toarray())
# print(len(after_one_hot))
# ##################################################
# #                  SAVE FILE                     #
# ##################################################
# with open('after_one_hot1','wb') as file:
#     pickle.dump(after_one_hot, file, protocol = 2)
# =======
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle
##################################################
#                  LOAD DATA                     #
##################################################
before_one_hot = pd.read_csv('preprocessed_data_for_one_hot.csv')
##################################################
#                   FUNCTION                     #
##################################################
enc = OneHotEncoder()
temp=[]
after_one_hot = []
accountcode_number = len(set(before_one_hot['accountcode']))
cardverificationcodesupplied_number = len(set(before_one_hot['cardverificationcodesupplied']))
cvcresponsecode_number = len(set(before_one_hot['cvcresponsecode']))
issuercountrycode_number = len(set(before_one_hot['issuercountrycode']))
shoppercountrycode_number = len(set(before_one_hot['shoppercountrycode']))
shopperinteraction_number = len(set(before_one_hot['shopperinteraction']))
txvariantcode_number = len(set(before_one_hot['txvariantcode']))
print(accountcode_number, cardverificationcodesupplied_number,cvcresponsecode_number,
     issuercountrycode_number,shoppercountrycode_number,shopperinteraction_number, txvariantcode_number,
     accountcode_number+cardverificationcodesupplied_number+cvcresponsecode_number+
     issuercountrycode_number+shoppercountrycode_number+shopperinteraction_number+txvariantcode_number)
##################################################
#                   ONE HOT                      #
##################################################
for row in before_one_hot.iterrows():
    index, data = row
    if index ==0:
        continue
    if data['cvcresponsecode'] >= 3:
        data['cvcresponsecode'] == 3
    temp.append(data.tolist())
    print(index)
enc.fit(temp)
print ("enc.n_values_ is:",enc.n_values_)
print ("enc.feature_indices_ is:",enc.feature_indices_)

for row in before_one_hot.iterrows():
    index, data = row
    if index ==0:
        continue
    print(index)
    my_data = (np.asarray(data)).reshape(1, -1)
    after_one_hot.append(enc.transform(my_data).toarray())
##################################################
#                  SAVE FILE                     #
##################################################
with open('after_one_hot1','wb') as file:
    pickle.dump(after_one_hot, file, protocol = 2)
