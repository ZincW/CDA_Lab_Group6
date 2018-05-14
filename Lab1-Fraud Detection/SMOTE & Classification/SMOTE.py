import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from collections import Counter
from sklearn import metrics

src='after_normalization.csv'
df=pd.read_csv(src)
df=df[~df['mail_id'].isnull()]
data_train, data_test, label_train, label_test = train_test_split(df, df['label'], test_size = 0.3)#test_size: proportion of train/test data
del data_train['label']
del data_test['label']
#print(data_train.iloc[1])
#-----------------------
print('Label number of training set:')
print(Counter(label_train))
print('Label number of test set:')
print(Counter(label_test ))
sm=SMOTE(random_state=20)
data_train_smote,label_train_smote=sm.fit_sample(data_train,label_train)
print('Label number of SMOTEd training set ')
print(Counter(label_train_smote))


##################################################
#                Random Forest                   #
##################################################
print('##################################################')
print('#                Random Forest                   #')
print('##################################################')
RF=RandomForestClassifier(n_estimators=50 , n_jobs=2)
RF.fit(data_train_smote,label_train_smote)
predict_label=RF.predict(data_test)
print('SMOTEd data:')
print('Differences between predicted label and true label:')
print(' 0:the number of data classified correctly')
print(' 1:the number of false positive data')
print('-1:the number of false nagatve data')
print(Counter(predict_label-label_test))
print(' ')
fpr_RF=dict()
tpr_RF=dict()
roc_auc_RF=dict()
probability_predict_RF=dict()
probability_predict_RF['smote']=RF.predict_proba(data_test)
fpr_RF['smote'],tpr_RF['smote'],thresholds=metrics.roc_curve(label_test,probability_predict_RF['smote'][:,1],pos_label=1)
roc_auc_RF['smote']=metrics.auc(fpr_RF['smote'],tpr_RF['smote'])


RF.fit(data_train,label_train)
probability_predict_RF['origin']=RF.predict_proba(data_test)
predict_label=RF.predict(data_test)
print('UNSMOTEd data:')
print('Differences between predicted label and true label:')
print(' 0:the number of data classified correctly')
print(' 1:the number of false positive data')
print('-1:the number of false nagatve data')
print(Counter(predict_label-label_test))
fpr_RF['origin'],tpr_RF['origin'],thresholds1=metrics.roc_curve(label_test,probability_predict_RF['origin'][:,1],pos_label=1)
roc_auc_RF['origin']=metrics.auc(fpr_RF['origin'],tpr_RF['origin'])

plt.title('Receiver Operating Characteristic - %s' % 'Random Forest')

plt.plot(fpr_RF['origin'], tpr_RF['origin'], 'g', label='UNSMOTE AUC = %0.2f' % roc_auc_RF['origin'])
plt.plot(fpr_RF['smote'], tpr_RF['smote'], 'b', label='SMOTE AUC = %0.2f' % roc_auc_RF['smote'])
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

##################################################
#                Logistic Regression             #
##################################################
print('##################################################')
print('#                Logistic Regression             #')
print('##################################################')
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(C=10)
LR.fit(data_train_smote,label_train_smote)
print('SMOTEd data:')
print('Differences between predicted label and true label:')
print(' 0:the number of data classified correctly')
print(' 1:the number of false positive data')
print('-1:the number of false nagatve data')
print(Counter(predict_label-label_test))
fpr_LR=dict()
tpr_LR=dict()
roc_auc_LR=dict()
probability_predict_LR=dict()
probability_predict_LR['smote']=LR.predict_proba(data_test)
fpr_LR['smote'],tpr_LR['smote'],thresholds=metrics.roc_curve(label_test,probability_predict_LR['smote'][:,1],pos_label=1)
roc_auc_LR['smote']=metrics.auc(fpr_LR['smote'],tpr_LR['smote'])
print('')
LR.fit(data_train,label_train)
probability_predict_LR['origin']=LR.predict_proba(data_test)
predict_label=LR.predict(data_test)
print('UNSMOTEd data:')
print('Differences between predicted label and true label:')
print(' 0:the number of data classified correctly')
print(' 1:the number of false positive data')
print('-1:the number of false nagatve data')
print(Counter(predict_label-label_test))
fpr_LR['origin'],tpr_LR['origin'],thresholds1=metrics.roc_curve(label_test,probability_predict_LR['origin'][:,1],pos_label=1)
roc_auc_LR['origin']=metrics.auc(fpr_LR['origin'],tpr_LR['origin'])

plt.title('Receiver Operating Characteristic - %s' % 'Logistic Regression')
plt.plot(fpr_LR['origin'], tpr_LR['origin'], 'g', label='UNSMOTE AUC = %0.2f' % roc_auc_LR['origin'])
plt.plot(fpr_LR['smote'], tpr_LR['smote'], 'b', label='SMOTE AUC = %0.2f' % roc_auc_LR['smote'])
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

##################################################
#                    Naive Bayes                 #
##################################################
print('##################################################')
print('#                    Naive Bayes                 #')
print('##################################################')
from sklearn.naive_bayes import GaussianNB
GNB=GaussianNB()
GNB.fit(data_train_smote,label_train_smote)
predict_label=GNB.predict(data_test)
print('SMOTEd data:')
print('Differences between predicted label and true label:')
print(' 0:the number of data classified correctly')
print(' 1:the number of false positive data')
print('-1:the number of false nagatve data')
print(Counter(predict_label-label_test))
fpr_GNB=dict()
tpr_GNB=dict()
roc_auc_GNB=dict()
probability_predict_GNB=dict()
probability_predict_GNB['smote']=GNB.predict_proba(data_test)
fpr_GNB['smote'],tpr_GNB['smote'],thresholds=metrics.roc_curve(label_test,probability_predict_GNB['smote'][:,1],pos_label=1)
roc_auc_GNB['smote']=metrics.auc(fpr_GNB['smote'],tpr_GNB['smote'])
print('')
GNB.fit(data_train,label_train)
probability_predict_GNB['origin']=GNB.predict_proba(data_test)
predict_label=GNB.predict(data_test)
print('UNSMOTEd data:')
print('Differences between predicted label and true label:')
print(' 0:the number of data classified correctly')
print(' 1:the number of false positive data')
print('-1:the number of false nagatve data')
print(Counter(predict_label-label_test))
fpr_GNB['origin'],tpr_GNB['origin'],thresholds1=metrics.roc_curve(label_test,probability_predict_GNB['origin'][:,1],pos_label=1)
roc_auc_GNB['origin']=metrics.auc(fpr_GNB['origin'],tpr_GNB['origin'])
plt.title('Receiver Operating Characteristic - %s' % 'Naive Bayes')

plt.plot(fpr_GNB['origin'], tpr_GNB['origin'], 'g', label='UNSMOTE AUC = %0.2f' % roc_auc_GNB['origin'])
plt.plot(fpr_GNB['smote'], tpr_GNB['smote'], 'b', label='SMOTE AUC = %0.2f' % roc_auc_GNB['smote'])
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

