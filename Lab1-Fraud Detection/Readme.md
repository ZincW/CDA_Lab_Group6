# Lab 1 Fraud Detection

## Requirement
The code requires Python3 as well as the following python libraries:  

* pandas
* sklearn
* collections
* matplotlib
* imblearn
* currency_converter

## Usage
## Pre-processing
There are several steps in our pre-processing. To save your time, we provide our processed data and u can download or open it in python from here:
https://drive.google.com/file/d/1oyD9BAqcXM5sd6J0qctmSXPwdrOHk0Yl/view?usp=sharing
Using this 'after_normailization.csv', you can running code of SMOTE, white-box, black-box classifers.

If you want to process data from scratch, you can follow steps here:
1. running preprocess_test.ipynb in Visualization part to get 'preprocessed_data.csv'.
   In this code, we do these things:
    1) convert all countrycode into 4 digits
    2) delete data where 'simple_journal' is 'Refused'
    3) delete all rows which include NAN values
    4) convert all currency into euro.
       to import packages, you can follow instructions here: https://github.com/alexprengere/currencyconverter
    5) label all data
    6) apply integers to represent categorical variables (for example, if there is 'Ecommerce' in 'shopperinteraction', we set 1 for 'shopperinteraction', else we set 0)
    7) convert timestamp into digits
    8) save processed file as 'preprocessed_data.csv'
2. running one_hot.py in Visualization part to get 'after_one_hot.csv'. The input is 'preprocessed_data_for_one_hot.csv' which includes only categorical data. In this code, we apply one_hot for all categorical variables. U can also get access to 'after_one_hot.csv' from here: https://drive.google.com/file/d/17JZg2K4WEDtQo0Hi3eCk0-gQElmdhg9D/view?usp=sharing
3. running normalization.py in SMOTE&Classification part to get our final processed data called 'after_normailization.csv'.
### Visualization
1. Running `preprocess_visulization` to preprocess the original data(`data_for_student_case.csv`) and get the `preprocessed_visdata.csv`.
2. Running `visualization` to see the plots. 

### SMOTE
1. Running `SMOTE` to see the performance of *Random Forest*, *Logistic Regression* and *Naive Bayes* of SMOTEd and UNSOMTEd data.

### Classification
1. Running `blackbox` in SMOTE&Classification part to see the performance of black-box algorithm that we choose.
2. Running `KNN.ipynb` in SMOTE&Classification part to see the performance of black-box algorithm that we choose.
   if you want to check our try in neural network, please click neural_network.py
### Bonus 
1. related code can be found in bonus.ipynb
### Contact
If you have any question about code, feel free to contact: Z.Wang-17@student.tudelft.nl
