import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 16)
pd.set_option('precision', 2)

import warnings
warnings.filterwarnings(action='ignore')

dataset='Data_Train.xlsx'
Train=pd.read_excel(dataset)
dataset1='Data_Test.xlsx'
Test=pd.read_excel(dataset)
Test_output=Test.copy()
'''
Name                  object
Location              object
Year                   int64
Kilometers_Driven      int64
Fuel_Type             object
Transmission          object
Owner_Type            object
Mileage               object
Engine                object
Power                 object
Seats                float64
New_Price             object
Price                float64
dtype: object
'''
Train=Train.drop('Name',axis=1)
Test=Test.drop('Name',axis=1)
Train=Train.drop('Location',axis=1)
Test=Test.drop('Location',axis=1)
Train=Train.drop('New_Price',axis=1)
Test=Test.drop('New_Price',axis=1)
'''
Cleaning Tranmission
Hence we can solve this as 
taking 
Manual = 0
Automatic = 1
'''
'''
Cleaning 
Hence we can solve this as 
taking 
Manual = 0
Automatic = 1
'''
import datetime
now=datetime.datetime.now()
now1=now.year
Train['Year']=(Train['Year']-now1)*(-1)
Test['Year']=(Test['Year']-now1)*(-1)
import re
def remove(list):
    pattern = '[A-z]'
    patt = '[/]'
    pattn= '[ ]'

    list = [re.sub(pattern, '', i) for i in list]
    list = [re.sub(patt,'',i) for i in list]
    list = [re.sub(pattn,'',i) for i in list]
    list = [re.sub(r' km/kg','',i) for i in list]
    return list

Train['Mileage']=Train['Mileage'].fillna('0')
Train['Power']=Train['Power'].fillna('0')
Train['Engine']=Train['Engine'].fillna('0')
Power1 = remove(Train['Power'])
Engine1 = remove(Train['Engine'])
Mileage1 = remove(Train['Mileage'])
Power1 = pd.to_numeric(Power1,errors='coerce')
Engine1 = pd.to_numeric(Engine1,errors='coerce')
Mileage1 = pd.to_numeric(Mileage1,errors='coerce')
print('The data type of Mileage 1 is :', Mileage1.dtype)
print('The data type of Power1 is :', Power1.dtype)
print('The data type of Engine 1 is :', Engine1.dtype)
Train['Seats']=Train['Seats'].fillna(0)
Train['Mileage1']=Mileage1
Train['Engine1']=Engine1
Train['Power1']=Power1

Test['Mileage']=Test['Mileage'].fillna('0')
Test['Power']=Test['Power'].fillna('0')
Test['Engine']=Test['Engine'].fillna('0')
Power1 = remove(Test['Power'])
Engine1 = remove(Test['Engine'])
Mileage1 = remove(Test['Mileage'])
Power1 = pd.to_numeric(Power1,errors='coerce')
Engine1 = pd.to_numeric(Engine1,errors='coerce')
Mileage1 = pd.to_numeric(Mileage1,errors='coerce')
print('The data type of Mileage 1 is :', Mileage1.dtype)
print('The data type of Power1 is :', Power1.dtype)
print('The data type of Engine 1 is :', Engine1.dtype)
Test['Seats']=Test['Seats'].fillna(0)
Test['Mileage1']=Mileage1
Test['Engine1']=Engine1
Test['Power1']=Power1

Train=Train.drop(['Mileage','Engine','Power'],axis=1)
Test=Test.drop(['Mileage','Engine','Power'],axis=1)

Owner1 = {"First": 1, "Second": 2, "Third": 3, "Fourth & Above":4}
Train['Owner_Type'] = Train['Owner_Type'].map(Owner1)
Test['Owner_Type'] = Test['Owner_Type'].map(Owner1)

Fuel= {'CNG':1, 'Diesel':2, 'Petrol':3, 'LPG':4, 'Electric':5}
Train['Fuel_Type'] = Train['Fuel_Type'].map(Fuel)
Test['Fuel_Type'] = Test['Fuel_Type'].map(Fuel)

Trans = {'Manual':1, 'Automatic':2}
Train['Transmission'] = Train['Transmission'].map(Trans)
Test['Transmission'] = Test['Transmission'].map(Trans)
Test=Test.drop('Price',axis=1)

sbn.barplot(x="Year", y="Price", data=Train)
plt.show()

sbn.barplot(x="Fuel_Type", y="Price", data=Train)
plt.show()

sbn.barplot(x="Transmission", y="Price", data=Train)
plt.show()

sbn.barplot(x="Owner_Type", y="Price", data=Train)
plt.show()

sbn.barplot(x="Seats", y="Price", data=Train)
plt.show()

print("Train.describe()\n",Train.describe(include='all'))
print("Test.describe()\n",Test.describe(include='all'))
hnames=['Year',  'Kilometers_Driven',  'Fuel_Type',  'Transmission',  'Owner_Type',    'Seats',    'Price',  'Mileage1',  'Engine1',   'Power1']
correlations = Train.corr()
print( correlations  )

# plot correlation matrix
fig = plt.figure()

subFig = fig.add_subplot()

cax = subFig.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)

ticks = np.arange(0,10)   # It will generate values from 0....10
subFig.set_xticks(ticks)
subFig.set_yticks(ticks)
subFig.set_xticklabels(hnames)
subFig.set_yticklabels(hnames)

plt.show()

columns = np.full((correlations.shape[0],), True, dtype=bool)
for i in range(correlations.shape[0]):
    for j in range(i+1, correlations.shape[0]):
        if correlations.iloc[i,j] >= 0.3:
            if columns[j]:
                columns[j] = False

selected_columns = Train.columns[columns]
print("Selected Columns = ",selected_columns)
selected_columns1 = Train.columns[columns]
input_predictors = Train[selected_columns]
print(Train.dtypes)

ouptut_target=Train['Price']

x_test=Test[selected_columns1]

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val=train_test_split(
    input_predictors, ouptut_target, test_size = 0.5, random_state = 0)

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression as lr
reg = lr()
reg.fit(x_train, y_train)

y_pred=reg.predict(x_val)
print("mean_absolute_error",mae(y_val, y_pred))
print("mean_squared_error : ",mse(y_val, y_pred))
print("Sqrt of mean_squared_error",np.sqrt(mse(y_val, y_pred)))

y_test=reg.predict(x_test)
print('Y predicted value is :')
print(y_test)


Test_output['Price']=y_test
import xlsxwriter
Test_output.to_excel('output.xlsx', engine='xlsxwriter')