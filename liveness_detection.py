#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#import tensorflow


# In[2]:


from keras.models import Sequential
from keras.layers import Dense
from keras import utils


# In[3]:


df1 = pd.read_csv("Real_model2_exposure.csv")
df2 = pd.read_csv("Fake_model2_exposure.csv")
print(df1.head(),"\n",df2.head())
df1.shape


# In[4]:


df1.drop(df1.columns[0], axis=1, inplace=True)
df2.drop(df2.columns[0], axis=1, inplace=True)
df1


# In[9]:


mean_real = []
mean_fake = []
for col in df1.columns:
    mean_real.append(df1[col].mean())
    mean_fake.append(df2[col].mean())
    df1[col] = df1[col].replace(np.NaN, mean_real[-1])
    df2[col] = df2[col].replace(np.NaN, mean_fake[-1])


# In[10]:


realX = np.array(df1)
fakeX = np.array(df2)
print(realX.shape, fakeX.shape)

np.random.shuffle(realX)
np.random.shuffle(fakeX)


# In[7]:


# real_size = int(0.80 * len(realX))
# fake_size = int(0.80 * len(fakeX))
# print(realX.shape, fakeX.shape)

# X_train = np.concatenate((realX[:real_size], fakeX[:fake_size]), axis=0)
# X_test = np.concatenate((realX[real_size:], fakeX[fake_size:]), axis=0)

# Y_train = np.array([1]*real_size + [0]*fake_size)
# Y_test = np.array([1]*(len(realX)-real_size) + [0]*(len(fakeX)-fake_size))

# print(X_train.shape, Y_train.shape, "\n", X_test.shape, Y_test.shape, "\n", Y_train, "\n", Y_test)


# In[8]:


# X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# print(X_train.shape, X_test.shape)


# In[10]:


#print(X_train[0][0])
# X_train = utils.normalize(X_train, axis=2)
# X_test = utils.normalize(X_test, axis=2)
#print(X_train[0][0])


# In[11]:


X = np.concatenate((realX,fakeX), axis=0)
Y = np.array([1]*len(realX) + [0]*len(fakeX))


# In[19]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[20]:


model = Sequential()
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[21]:


history = model.fit(X, Y, validation_split=0.2, epochs=2500, shuffle=True)


# In[37]:


# plt.plot(history.history["loss"])
# plt.xlabel("epoch")
# plt.ylabel("cross_entropy_loss")
# plt.show()

# fig = plt.figure(figsize=(20,10))
# plt.plot(history.history["accuracy"])
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.show()


# In[22]:


figure = plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label='loss')
plt.plot(history.history["val_loss"], label='val loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

figure = plt.figure(figsize=(10,5))
plt.plot(history.history["accuracy"], label='accuracy')
plt.plot(history.history["val_accuracy"], label='val accuracy')
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()


# In[38]:


# val_loss, val_accuracy = model.evaluate(X_test, Y_test)
# print(val_loss, val_accuracy)


# In[25]:


df3 = pd.read_csv("TestReal_model2_exposure.csv")
df4 = pd.read_csv("TestFake_model2_exposure.csv")

df3.drop(df3.columns[0], axis=1, inplace=True)
df4.drop(df4.columns[0], axis=1, inplace=True)

# df3 = df3.fillna(0)
# df4 = df4.fillna(0)
# print(df3,"\n",df4)

for col in df3.columns:
    df3[col] = df3[col].replace(np.NaN, mean_real[int(col)])
    df4[col] = df4[col].replace(np.NaN, mean_fake[int(col)])


realX_test = np.array(df3)
fakeX_test = np.array(df4)
print(realX_test.shape, fakeX_test.shape)

# np.random.shuffle(realX_test)
# np.random.shuffle(fakeX_test)


# In[26]:


X_test = np.concatenate((realX_test,fakeX_test), axis=0)
Y_test = np.array([1]*len(realX_test) + [0]*len(fakeX_test))

X_test = scaler.transform(X_test)
print(X_test.shape, Y_test.shape)


# In[27]:


Y_test_pred = model.predict(X_test)
print(Y_test_pred)


# In[28]:


model.evaluate(X_test, Y_test)


# In[31]:


model.save("single_hidden_layer_updated.model")


# In[ ]:




