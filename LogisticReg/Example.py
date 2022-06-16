import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

train = pd.read_csv('LogisticReg/titanic_train.csv') 

print(train.head()) 
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') 

sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Pclass', data=train) 

# sns.distplot(train['Age'].dropna(),kde=False,bins=100)

#sns.countplot(x='SibSp',data=train)

# train['Fare'].hist(bins=40,figsize=(10,4))

# plt.show()  

# import cufflinks as cf  
# cf.go_offline() 
# train['Fare'].iplot(kind='hist',bins=30) 

plt.figure(figsize=(10,7))
sns.boxenplot(x='Pclass',y='Age',data=train) 

plt.show()  

def impute_age(cols): 
    Age = cols[0] 
    Pclass = cols[1]

    if pd.insull(Age): 
        if Pclass == 1: 
            return 37 
        elif Pclass == 2: 
            return 29  
        else: 
            return Age

