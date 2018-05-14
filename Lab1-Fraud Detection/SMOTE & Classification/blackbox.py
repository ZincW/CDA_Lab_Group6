import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score

#read data
src='after_normalization.csv'
df=pd.read_csv(src)
#df = df.dropna(axis=0, how='any')

#obtain training set and test set
data_train, data_test, label_train, label_test = train_test_split(df, df['label'], test_size = 0.35)#test_size: proportion of train/test data
del data_train['label']
del data_test['label']
#print(data_train.ix[15627,:])

print('------dataset------')
print('Label number of training set:')
print(Counter(label_train))
print('Label number of test set:')
print(Counter(label_test ))

#Random Forest
RF=RandomForestClassifier(n_estimators=50 , n_jobs=2)

#cross-validation
print('------cross validation------')
scores = cross_val_score(RF, data_train, label_train, cv=10,scoring='accuracy')
#print(scores)
plt.plot(1-scores)
plt.ylabel('Misclassification Error')
plt.xlabel('Times of 10-fold cross-validation')
plt.show()


print('------classification------')	
RF.fit(data_train,label_train)
predict_label=RF.predict(data_test)

print('Differences between predicted label and true label:')
print(' 0:the number of data classified correctly')
print(' 1:the number of false positive data')
print('-1:the number of false nagatve data')
print(Counter(predict_label-label_test))
print(' ')

tn, fp, fn, tp = confusion_matrix(label_test, predict_label).ravel()
#print('tp=',tp,' fp=',fp,' fn=',fn,' tn=',tn)
print('Recall=',tp/(tp+fn))
print('FPR=',fp/(fp+tn))

#ROC-curve
probability_predict_RF=RF.predict_proba(data_test)
fpr_RF,tpr_RF,thresholds=metrics.roc_curve(label_test,probability_predict_RF[:,1],pos_label=1)
roc_auc_RF=metrics.auc(fpr_RF,tpr_RF)		
plt.plot(fpr_RF, tpr_RF ,'b', label='AUC = %0.2f' % roc_auc_RF)
plt.title('Receiver Operating Characteristic - %s' % 'Blackbox Algorithm')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
