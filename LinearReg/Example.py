import pandas as pd
import numpy as np 


import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('USA_Housing.csv')  
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']] 
y = df['Price'] 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)  

from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 
lm.fit(X_train,y_train)  

cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coef'])

print(cdf) 

predictions = lm.predict(X_test) 
plt.scatter(y_test, predictions) 
plt.show() 

sns.displot((y_test - predictions)) 
plt.show() 

from sklearn import metrics 
print(
metrics.mean_absolute_error(y_test, predictions),
metrics.mean_squared_error(y_test, predictions),
np.sqrt(metrics.mean_squared_error(y_test, predictions)) 
)
