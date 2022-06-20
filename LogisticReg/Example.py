from operator import concat
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

train = pd.read_csv('LogisticReg/titanic_train.csv') 
# print(train.head()) 

#~~~~~This is a bunch of visualization nonsense. 
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') 
# plt.show()

# sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Pclass', data=train) 

# sns.distplot(train['Age'].dropna(),kde=False,bins=100)

#sns.countplot(x='SibSp',data=train)

# train['Fare'].hist(bins=40,figsize=(10,4))

# plt.show()  

# import cufflinks as cf  
# cf.go_offline() 
# train['Fare'].iplot(kind='hist',bins=30) 

# plt.figure(figsize=(10,7))
# sns.boxenplot(x='Pclass',y='Age',data=train) 

# plt.show()  

#~~~~~~DATA CLEANING - Important to clean incoming data for an ML algorithm. 
#~~~~~This block will interpolate a passengers age if none is listed.
def impute_age(cols): 
    Age = cols[0] 
    Pclass = cols[1]

    if pd.isnull(Age): 
        if Pclass == 1: 
            return 37 
        elif Pclass == 2: 
            return 29  
        else: 
            return 24 
    else: 
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1) 

#~~~~~This block will simply remove the Cabin column as it had too many null values. 
train.drop('Cabin', axis=1,inplace=True) 
# print(train.head())
train.dropna(inplace=True) 

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis') 
# plt.show()   

#~~~~~What this si doing is replacing columns with simplified versions.
# for instance once that we want to avoid is redundancy in the the ability of one colmun 
# prodicting the value of another column perfectly and visavera. 
# So we remove one of the columns as one of them can hold the same value. 
sex = pd.get_dummies(train['Sex'],drop_first=True) 
embark = pd.get_dummies(train['Embarked'],drop_first=True)
pclass = pd.get_dummies(train['Pclass'],drop_first=False)
train = pd.concat([train,sex,embark,pclass],axis=1) 


#Dropping all redundant and not useful columns and data.
train.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)

#~~~~~Not needed as we already have row index.
train.drop(['PassengerId'],axis=1,inplace=True)
print(train.head()) 

#~~~~~Now that our data is cleaned we are going to run it through ML. 
#~~~~~~Need to start with spliting our data into TRAINING,VALIDATION, AND TESTING sets. 
X = train.drop('Survived',axis=1)
y = train['Survived']  

from sklearn.model_selection import train_test_split   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression 
logmodel = LogisticRegression() 
logmodel.fit(X_train,y_train) 

predictions = logmodel.predict(X_test) 


from sklearn.metrics import classification_report 
print(classification_report(y_test,predictions)) 

from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test,predictions))