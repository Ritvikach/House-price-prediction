#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import seaborn as sns


# In[2]:


df=pd.read_csv('Bengaluru_House_Data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


df.groupby('area_type')['area_type'].count()


# In[9]:


df2=df.drop(['area_type','society'],axis='columns')


# In[10]:


df2.head()


# In[11]:


df2.isnull().sum()


# In[12]:


df2.groupby('balcony')['balcony'].count()


# In[13]:


sns.pairplot(df2)


# In[14]:


df.corr()


# In[15]:


df2=df2.drop('balcony',axis='columns')


# In[16]:


df2.groupby('availability').count()


# In[17]:


df2=df2.drop('availability',axis='columns')


# In[18]:


df2.info()


# In[19]:


df3=df2.dropna()


# In[20]:


df3.isnull().sum()


# In[21]:


df3.shape


# In[22]:


df3['size'].unique()


# In[23]:


df3['rooms']=df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[24]:


df3.head()


# In[25]:


df3['rooms'].unique()


# In[26]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[27]:


df3[~df3['total_sqft'].apply(is_float)].tail(10)


# In[28]:


def convert_to_num(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[29]:


df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_to_num)


# In[30]:


df4.isnull().sum()


# In[31]:


# price per squarefeet


# In[32]:


df5=df4.copy()


# In[33]:


df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']


# In[34]:


df5.head()


# In[35]:


len(df5['location'].unique())


# In[36]:


df5['location']=df5['location'].apply(lambda x:x.strip())


# In[37]:


location_stat=df5.groupby('location')['location'].agg('count')


# In[38]:


location_stat


# In[39]:


location_stat_less10=location_stat[location_stat<=10]


# In[40]:


df5.location=df5.location.apply(lambda x:'other' if x in location_stat_less10 else x)


# In[41]:


len(df5['location'].unique())


# In[42]:


df5[df5['total_sqft']/df5['rooms']<150].count()


# In[43]:


df5.shape


# In[44]:


df6=df5[~(df5['total_sqft']/df5['rooms']<300)]


# In[45]:


df6.shape


# In[46]:


df6['price_per_sqft'].describe()


# In[47]:


sns.histplot(df6['price_per_sqft'],kde=True)


# In[48]:


def remove_outliers(df):
    df_final=pd.DataFrame()
    for key , subdf in df.groupby('location'):
        Q1 = subdf['price_per_sqft'].quantile(0.25)
        Q3 = subdf['price_per_sqft'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_subdf = subdf[(subdf['price_per_sqft'] >= lower_bound) & (subdf['price_per_sqft'] <= upper_bound)]
        df_final = pd.concat([df_final, filtered_subdf], ignore_index=True)
    return df_final     


# In[49]:


df7=remove_outliers(df6)


# In[50]:


df7.shape


# In[51]:


df7.columns


# In[52]:


def plot_bhk_prices(df, location):
    
    location_df = df[df['location'] == location]
    
    plt.figure(figsize=(10, 6))
    
    # Plot for 2 BHK
    bhk_2 = location_df[location_df['rooms'] == 2]
    plt.scatter(bhk_2['price_per_sqft'], bhk_2['price'], color='blue', label='2 BHK')
    
    # Plot for 3 BHK
    bhk_3 = location_df[location_df['rooms'] == 3]
    plt.scatter(bhk_3['price_per_sqft'], bhk_3['price'], color='green', label='3 BHK')
    
    plt.xlabel('Price per Square Foot')
    plt.ylabel('Price')
    plt.title(f'2 BHK vs 3 BHK Prices in {location}')
    plt.legend()
    plt.show()


# In[53]:


df7['location'].unique()


# In[54]:


plot_bhk_prices(df7,'Rajaji Nagar')


# In[55]:


plot_bhk_prices(df7,'Kasavanhalli')


# In[56]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        
        for rooms, bhk_df in location_df.groupby('rooms'):
            bhk_stats[rooms] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        
        for rooms, bhk_df in location_df.groupby('rooms'):
            stats = bhk_stats.get(rooms - 1)
            if stats and stats['count'] > 5:
                indices_to_exclude = bhk_df[bhk_df['price_per_sqft'] < stats['mean']].index.values
                exclude_indices = np.append(exclude_indices, indices_to_exclude)
    
    return df.drop(exclude_indices, axis='index')


# In[57]:


df8=remove_bhk_outliers(df7)


# In[58]:


df8.shape


# In[59]:


sns.histplot(df8['price_per_sqft'],bins=50)


# In[60]:


df8['bath'].unique()


# In[61]:


df8[df8['bath']>10]


# In[62]:


df9=df8[df8.bath<df8.rooms+2]


# In[63]:


df9.shape


# In[64]:


df10=df9.drop(['size','price_per_sqft'],axis='columns')


# In[65]:


df10.head()


# In[66]:


dummies=pd.get_dummies(df10['location']) # one hot encoding


# In[67]:


df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')


# In[68]:


df11.head()


# In[69]:


df12=df11.drop('location',axis='columns')
df12.head()


# In[70]:


df12.shape


# In[71]:


# droping the dependant variables that is price


# In[72]:


X=df12.drop('price',axis='columns')


# In[73]:


y=df12.price


# In[74]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


# In[75]:


from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(X_train,y_train)


# In[76]:


lr_clf.score(X_test,y_test)


# In[77]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


# In[78]:


cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)


# In[79]:


cross_val_score(LinearRegression(),X,y,cv=cv)


# In[80]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    
    scores = []
    
    for algo_name, config in algos.items():
        cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


# In[81]:


find_best_model_using_gridsearchcv(X, y)


# In[82]:


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]


# In[83]:


predict_price('Indira Nagar',1000,2,2)


# In[85]:


import pickle
with open('home_price_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[86]:


import json
columns={'data_columns':[col.lower() for col in X.columns]}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




