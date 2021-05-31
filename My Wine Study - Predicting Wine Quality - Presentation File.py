#!/usr/bin/env python
# coding: utf-8

# # Vinho Verde - Wine Dataset Predicting Wine Quality 
# ### My dataframe - Wines
# #### Machine Learning - Predicting Wine Type (High, Low, Medium)

# In[1]:


# Set up my wine dataset - My DataFrame

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# In[2]:


RW_df = pd.read_csv('winequality-red.csv', delimiter = ';')
WW_df = pd.read_csv('winequality-white.csv', delimiter = ';')


# In[3]:


WW_df['type'] = 'White Wine'
RW_df['type'] = 'Red Wine'


# In[4]:


df_wines = pd.concat([WW_df, RW_df])


# In[5]:


def label(x):
    if (x == 8) or (x == 9) or (x == 7)  :
        return 'High'
    
    elif x in [5, 6]:
        return 'Medium'
    else:
        return 'Low'


# In[6]:


df_wines['quality_label'] = df_wines['quality'].apply(lambda x: label(x))


# In[7]:


df_wines['quality_label'].value_counts()


# In[ ]:





# In[ ]:


# Quality levels:

                # Medium : 5 or 6 
               
               # High : 7, 8 or 9
        
              # Low : 4 or lower 


# In[ ]:





# ### Preprocessing

# In[8]:


wqp_features = df_wines.iloc[:,:-3] # X

wqp_class_labels = np.array(df_wines['quality_label']) # y

wqp_label_names = ['Low', 'Medium', 'High'] 

wqp_feature_names = list(wqp_features.columns)

# Splitting into train and test. We need to separate out the prediction class in train and test set. 

# Here the test size is 30%.

wqp_train_X, wqp_test_X, wqp_train_y, wqp_test_y = train_test_split(wqp_features,
wqp_class_labels, test_size=0.3, random_state=42)

print(Counter(wqp_train_y), Counter(wqp_test_y))

print('Features:', wqp_feature_names)


# In[9]:


# Define the scaler

wqp_ss = StandardScaler().fit(wqp_train_X)

# Scale the train set

wqp_train_SX = wqp_ss.transform(wqp_train_X)

# Scale the test set

wqp_test_SX = wqp_ss.transform(wqp_test_X)


# In[10]:


le = LabelEncoder() 

df_wines['quality_label'] = le.fit_transform(df_wines['quality_label'])


# In[11]:


df_wines['quality_label'].value_counts()


# In[12]:


# ---


# ### Modelling

# #### 1-) Decision Tree - Using the LMS codes

# In[13]:


# train the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report


# In[14]:



wqp_dt = DecisionTreeClassifier()

wqp_dt.fit(wqp_train_SX, wqp_train_y)


# In[15]:


# predict and evaluate performance

wqp_dt_predictions = wqp_dt.predict(wqp_test_SX)

print(classification_report(wqp_test_y,wqp_dt_predictions, target_names=wqp_label_names))


# In[16]:


# Cohen's Kappa 

from sklearn.metrics import cohen_kappa_score


# In[17]:


cohen_kappa_score(wqp_test_y,wqp_dt_predictions)


# In[18]:


# ---


# #### 2-) Random Forest

# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


wqp_rf = RandomForestClassifier(random_state=1)


# In[21]:


# Fitting the model

wqp_rf.fit(wqp_train_SX, wqp_train_y)


# In[22]:


# predict and evaluate performance

wqp_rf_predictions = wqp_rf.predict(wqp_test_SX)

print(classification_report(wqp_test_y,wqp_rf_predictions, target_names=wqp_label_names))


# In[23]:


cohen_kappa_score(wqp_test_y,wqp_rf_predictions)


# In[76]:


# ---


# #### 3-) Trying to implement AdaBoost

# In[96]:



from sklearn.ensemble import AdaBoostClassifier


# In[97]:


model3 = AdaBoostClassifier(random_state=1)


# In[98]:


model3.fit(wqp_train_SX, wqp_train_y)


# In[99]:


y_pred3 = model3.predict(wqp_test_SX)


# In[100]:


print(classification_report(wqp_test_y, y_pred3))


# In[95]:


# ---


# #### 4-) Trying to implement XGBoost

# In[87]:


import xgboost as xgb


# In[88]:


model4 = xgb.XGBClassifier(random_state=1)


# In[103]:


model4.fit(wqp_train_SX, wqp_train_y)

y_pred4 = model4.predict(wqp_test_SX)

print(classification_report(wqp_test_y, y_pred4))


# In[104]:


# ---


# In[ ]:




