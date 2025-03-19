#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


diabete_df = pd.read_csv('diabetes.csv')
diabete_df.head(5)


# In[3]:


diabete_df.shape


# In[4]:


diabete_df['Outcome'].value_counts()


# In[5]:


diabete_df.info()


# In[6]:


diabete_df.describe()


# In[7]:


diabete_df.groupby('Outcome').mean()


# In[8]:


X = diabete_df.drop('Outcome',axis=1)
y = diabete_df['Outcome']


# In[9]:


print(X)


# In[10]:


print(y)


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
Standard_data = scaler.fit_transform(X)
X = Standard_data


# In[12]:


X


# ## TRAIN MODEL

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=25, random_state=1)


# In[14]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()


# In[15]:


model = svm.SVC(kernel='linear')


# In[16]:


lg.fit(X_train,y_train)


# In[17]:


train_y_pred = lg.predict(X_train)
test_y_pred = lg.predict(X_test)


# In[18]:


print('Train set Accuracy :',accuracy_score(train_y_pred,y_train))
print('Test set Accuracy :',accuracy_score(test_y_pred,y_test))


# In[24]:


import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset (replace with actual diabetes dataset)
# Assuming dataset has 8 features (like PIMA diabetes dataset)
data = pd.read_csv("diabetes.csv")  # Load your dataset
X = data.iloc[:, :-1].values  # Features (Assuming last column is target)
y = data.iloc[:, -1].values   # Target (Diabetes or Not)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = SVC()
model.fit(X_train, y_train)

# Input data (Must have the same 8 features)
input_data = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])  # Keep it as a 2D array

# Standardize input data using the same scaler
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)

# Output result
if prediction[0] == 1:
    print("This person has diabetes.")
else:
    print("This person does not have diabetes.")


# In[25]:


diabete_df.head(100)


# In[ ]:




