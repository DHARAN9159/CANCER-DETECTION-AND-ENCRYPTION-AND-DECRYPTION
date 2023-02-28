import warnings

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

all_cancer = pd.read_csv("dataset/Global cancer incidence both sexes.csv")
print(all_cancer.head())
print("------------------")
female = pd.read_csv("dataset/Global cancer incidence in women.csv")
print(female.head())
print("-----------------")
male = pd.read_csv('dataset/Global cancer incidence in men.csv')
print(male.head())


def shape(dataframe):
    print("=" * 20, 'dataframe shape', "=" * 20)
    print(dataframe.shape)
    print('\n')


def describe(dataframe):
    print("=" * 20, 'dataframe describe', "=" * 20)
    print(dataframe.describe())
    print('\n')


def info(dataframe):
    print("=" * 20, 'dataframe info', "=" * 20)
    print(dataframe.info())
    print('\n')

shape(male)
describe(male)
info(male)


shape(female)
info(female)
describe(female)

f= female.drop(["Unnamed: 0" , "Rank"],1)
print(f.head())

male = male.drop(["Unnamed: 0" , "Rank"],1)
male.head()


x= male['New_cases_in_2020']
y= female["New_cases_in_2020"]
plt.figure()
plt.plot(x)
plt.plot(y)
plt.xlabel('male new case 2020')
plt.ylabel('female new case 2020')
plt.title('all cancers')
plt.show()


x= male['New_cases_in_2020']
y= female["New_cases_in_2020"]
plt.figure()
plt.hist(x)
plt.hist(y)
plt.xlabel('male new case 2020')
plt.ylabel('female new case 2020')
plt.title('all cancers')
plt.show()

plt.pie(x, labels=male ,  autopct = "%0.2f%%")
plt.title('male cancer')
plt.show()


plt.pie(y, labels=f , autopct = "%0.2f%%")
plt.title('female cancer')
plt.show()
