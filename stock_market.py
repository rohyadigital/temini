#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow


# In[2]:


import pandas_datareader.data as pdr


# In[3]:


import pandas as pd


# In[4]:


import datetime as dt


# In[5]:


df = pdr.DataReader('RELIANCE.NS','yahoo', start='2015-01-01', end='2020-12-31')


# In[6]:


df.head


# In[7]:


df.to_csv('RELIANCE.NS.csv')


# In[8]:


df = pd.read_csv('RELIANCE.NS.csv')


# In[9]:


df.head()


# In[13]:


df1 = df.reset_index()['Close']


# In[14]:


df1.shape


# In[15]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[16]:


import numpy as np


# In[17]:


### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[18]:


df1.shape


# In[19]:


print(df1)


# In[20]:


###splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[21]:


training_size,test_size


# In[22]:


import numpy
def create_dataset(dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX), numpy.array(dataY)


# In[23]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[24]:


print(X_train.shape), print(y_train.shape)


# In[25]:


print(X_test.shape), print(ytest.shape)


# In[26]:


#reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[28]:


model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[29]:


model.summary()


# In[30]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[31]:


import tensorflow as tf


# In[32]:


tf.__version__


# In[45]:


###Lets do the predicition and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[46]:


###Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[35]:


pip install -U scikit-learn --user


# In[47]:


import sklearn


# In[37]:


sklearn.show_versions()


# In[38]:


pip install -U scikit-learn scipy matplotlib --user


# In[39]:


import sys


# In[40]:


print(sys.path)


# In[41]:


sklearn.__version__


# In[48]:


###Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[49]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[51]:


##Plotting
#shift train prediction for plotting
look_back =100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot [:, :] =np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
#shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :]=numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
#plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show


# In[52]:


len(test_data)


# In[54]:


x_input=test_data[418:].reshape(1,-1)
x_input.shape


# In[55]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[59]:


#logic to find next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input{}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input= x_input.reshape((1,n_steps,1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day out {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
print(lst_output)


# In[60]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[61]:


import matplotlib.pyplot as plt


# In[62]:


len(df1)


# In[63]:


df3=df1.tolist()
df3.extend(lst_output)


# In[64]:


plt.plot(day_new, scaler.inverse_transform(df1[1378:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[67]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[900:])


# In[ ]:




