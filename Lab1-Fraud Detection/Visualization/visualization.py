import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_set=pd.read_csv('preprocessed_visdata.csv')

#figure 1
sns.boxplot(x='simple_journal',y='euro_amount',data=data_set[data_set['euro_amount'] <= 30000])
plt.show()

#figure 2
sns.barplot(x='shopperinteraction',y='euro_amount',hue='simple_journal',data=data_set)
plt.show()


#figure 3
def countryplot(dataset,labe11,type,filename):
    filtered_data=dataset[dataset['simple_journal']==type]
    aggregation_data=filtered_data.groupby([label1,'simple_journal','euro_amount']).size().reset_index(name='count')
    aggregation_data.info()
    
    x = list(dataset[label1][dataset.simple_journal == 'Chargeback'].unique())
    aggregation_data = aggregation_data[aggregation_data[label1].isin(x)]

    
    sns.stripplot(x=label1, y=data_set['euro_amount'],data=aggregation_data)

    plt.xlabel('frequency of non-fraud data')
    plt.ylabel('euro_amount')
    plt.savefig(filename)
    plt.show()

label1='shoppercountrycode'
type='Chargeback'
fame='3'
countryplot(data_set,label1,type,fame)
type='Settled'
countryplot(data_set,label1,type,fame)
