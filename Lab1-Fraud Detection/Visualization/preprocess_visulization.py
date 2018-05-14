from currency_converter import CurrencyConverter
from collections import defaultdict
import numpy as np
import pandas as pd
import time
import datetime
import string
##################################################
#                  LOAD DATA                     #
##################################################
bank_data = pd.read_csv('data_for_student_case.csv')
# print bank_data.iloc[2]
##################################################
#                   FUNCTION                     #
##################################################
def rec_dd():
    return defaultdict(rec_dd)
def convert_countrycode_into_number(countrycode):
    #data is too large. if u want to test the code, use list(set()) as following:
#     countrycode_list = list(set(countrycode))
    countrycode_list = list(countrycode)
    countrycode_number_dict = defaultdict(list)
    countrycode_number_list = []
    for index in range(0, len(countrycode_list)):
        for item in list(countrycode_list[index]):
            countrycode_number_dict[index].append(str((string.ascii_uppercase).index(item)).zfill(2))
    for index in countrycode_number_dict:
        countrycode_number_dict[index] = ''.join(countrycode_number_dict[index])
    countrycode_number_list = countrycode_number_dict.values()
    return countrycode_number_list
c = CurrencyConverter()
##################################################
#                 PRE-PROCESSING                 #
##################################################
bank_data = bank_data[bank_data['simple_journal'] != 'Refused']
#bank_data  = bank_data.dropna(axis=0, how='any')
# transfer data
simple_journal_list=[]
euro_currency = []
booking_timestamp_list = []
creation_timestamp_list = []
mail_id_list=[]
ip_id_list=[]
card_id_list=[]
label = []
credictcard_or_visa = []
shopperinteraction_list = []
cardverificationcodesupplied_list = []
accountcode_list = []
txid_list = list(bank_data['txid'])
bin_list = list(bank_data['bin'])
cvcresponsecode_list = []
for index, row in bank_data.iterrows():
    simple_journalitem=str(row.simple_journal)
    currency = str(row.currencycode)
    booking_time = row.bookingdate
    creation_time = row.creationdate
    mail_id=str(row.mail_id)[5:]
    ip_id=str(row.ip_id)[2:]
    card_id=str(row.card_id)[4:]
    cvcresponsecode=row.cvcresponsecode

    euro = c.convert(row.amount, currency,'EUR')
    booking_timestamp = time.mktime(datetime.datetime.strptime(booking_time, '%Y-%m-%d %H:%M:%S').timetuple())
    creation_timestamp = time.mktime(datetime.datetime.strptime(creation_time, '%Y-%m-%d %H:%M:%S').timetuple())
    euro_currency.append(euro)
    simple_journal_list.append(simple_journalitem)
    booking_timestamp_list.append(booking_timestamp)
    creation_timestamp_list.append(creation_timestamp)
    mail_id_list.append(mail_id)
    ip_id_list.append(ip_id)
    card_id_list.append(card_id)
    if row['simple_journal'] == 'Chargeback':
        label.append(1)
    else:
        label.append(0)
    if (cvcresponsecode>=3):
        cvcresponsecode_list.append(3)
    else:
        cvcresponsecode_list.append(cvcresponsecode)
    if row['cardverificationcodesupplied'] == 'True':
        cardverificationcodesupplied_list.append(1)
    else:
        cardverificationcodesupplied_list.append(0)
    print (index)
	
issuercountrycode_number_list = bank_data['issuercountrycode']
shoppercountrycode_number_list = bank_data['shoppercountrycode']
shopperinteraction_list=bank_data['shopperinteraction']
credictcard_or_visa=bank_data['txvariantcode']
accountcode_list=bank_data['accountcode']

# print(issuercountrycode_number_list, shoppercountrycode_number_list)
##################################################
#            CONSTRUCT OUR DATA                  #
##################################################
preprocessed_data = pd.DataFrame({
    'txid':txid_list,
    'bookingdate':booking_timestamp_list,
    'issuercountrycode':issuercountrycode_number_list,
    'txvariantcode':credictcard_or_visa,
    'bin':bin_list,
    'euro_amount':euro_currency,
    'shoppercountrycode':shoppercountrycode_number_list,
    'shopperinteraction':shopperinteraction_list,
    'cardverificationcodesupplied':cardverificationcodesupplied_list,
    'cvcresponsecode':cvcresponsecode_list,
    'creationdate':creation_timestamp_list,
    'accountcode':accountcode_list,
    'mail_id': mail_id_list,
    'ip_id': ip_id_list,
    'card_id': card_id_list,
    'label':label,
    'simple_journal':simple_journal_list
    })
preprocessed_data  = preprocessed_data.dropna(axis=0, how='any')
##################################################
#                 SAVE AS CSV                    #
##################################################
preprocessed_data.to_csv('preprocessed_visdata.csv', encoding='utf-8', index=False)
