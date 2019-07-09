import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline


#Exploratory Data Analysis
data = pd.read_csv('../indian_liver_patient.csv')
print ('Head', data.head())

print ('Info', data.info())

data.groupby('Dataset').mean()

#dataset 1 is a patient with liver disease, 2 with not
#Albumin and Globulin is the only column with missing data. Because we only have 4 missing rows and
#the difference is fairly small between the two datasets, I am going to fill in the missing values with
#the total average
#Bilirubin and the Aminotransferases seem to be strong factors

data.groupby(data.Albumin_and_Globulin_Ratio.isnull()).mean()

There may significance in the missing data but because its only four values we will still take the average

print ('Desrcibe',data.describe())

#Lets change Gender to a continuous variable
#Male: 0 Female: 1

gender_num = {'Male': 0, 'Female':1}
data.Gender = data.Gender.map(gender_num)

#Overlaid histograms
#Age may not be the best indicator

for i in ['Albumin_and_Globulin_Ratio', 'Total_Bilirubin','Age']:
    sick = list(data[data['Dataset'] == 1][i].dropna()) # non missing values
    healthy = list(data[data['Dataset'] == 2][i].dropna())
    xmin = min(min(sick), min(healthy))
    xmax = max(max(sick), max(healthy))
    width = (xmax - xmin) / 40
    sns.distplot(sick, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(healthy, color='b', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['Liver Disease', 'Healthy'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()

#Fill missing values for Albumin and Globulin Ratio
data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean(), inplace = True)

#Check missing data
print (data.isnull().sum())

#writing clean data
data.to_csv('../liver_clean.csv', index = False)
clean = pd.read_csv('../liver_clean.csv')
print ('Clean Data Head', clean.head())

#Spliting Data into training, validation and test
#Split will be 60% training, 20% validation, 20% test

from sklearn.model_selection import train_test_split
features = clean.drop('Dataset', axis = 1)
labels = clean.Dataset


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.4, random_state=1)
test_features, val_features, test_labels, val_labels = train_test_split(test_features, test_labels, test_size=0.5, random_state=1)

#checking sizes
names = ['Training', 'Validation','Testing']
for i in [train_labels, val_labels, test_labels]:
    print ('{} Percentage: {}%'.format(names[i], round(len(i)*100 / len(labels),1))
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#First Cross validation model without tuning hyper parameters:

rf = RandomForestClassifier()
score = cross_val_score(rf, train_features, train_labels.values.ravel(), cv=5) #cv is number of folds

print('Accuracy Range: {}'.format(score))

#Range of 67% to 77% accuracy. Tuning the hyperparameters with grid search might enable us to reach the higher side

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [5, 50, 100], 'max_depth': [2, 10, 20, None]}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(train_features, train_labels.values.ravel())

def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

print_results(cv)

#Top 3 are : ('max_depth': 2, 'n_estimators': 100), ('max_depth': 2, 'n_estimators': 50) and ('max_depth': 20, 'n_estimators': 50)
#We will test these three on the validation set

rf1 = RandomForestClassifier(n_estimators=100, max_depth=2)
rf1.fit(train_features, train_labels.values.ravel())

rf2 = RandomForestClassifier(n_estimators=50, max_depth=2)
rf2.fit(train_features, train_labels.values.ravel())

rf3 = RandomForestClassifier(n_estimators=50, max_depth=20)
rf3.fit(train_features, train_labels.values.ravel())

#Accuracy, Precision and Recall

from sklearn.metrics import accuracy_score, precision_score, recall_score

for mdl in [rf1, rf2, rf3]:
    y_pred = mdl.predict(val_features)
    accuracy = round(accuracy_score(val_labels, y_pred), 3)
    precision = round(precision_score(val_labels, y_pred), 3)
    recall = round(recall_score(val_labels, y_pred), 3)
    print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(mdl.max_depth,
                                                                         mdl.n_estimators,
                                                                         accuracy,
                                                                         precision,
                                                                         recall))

#Max depth of 2 and 100 estimators seem to be the best hyperparameters

#To finish, we will test these hyperparamters on the test set

y_pred = rf1.predict(test_features)
accuracy = round(accuracy_score(test_labels,y_pred), 3)
precision = round(precision_score(test_labels, y_pred), 3)
recall = round(recall_score(test_labels, y_pred), 3)

print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(
                                rf2.max_depth, rf2.n_estimators, accuracy, precision, recall))

#These parameters preformed quite well on the test set

