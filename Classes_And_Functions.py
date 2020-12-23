#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer


# In[2]:


class AirbnbSeniorityCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self, data_compiled_date):
        self.date = data_compiled_date
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['Airbnb_seniority'] = X.apply(lambda x: self.date - x['host_since'], axis=1)
        X['Airbnb_seniority'] = X['Airbnb_seniority'].apply(lambda x: x.days)
        return X


# In[3]:


class AmenitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_amenitites = ['Microwave',
                                    'Stove',
                                    'Iron',
                                    'Free street parking',
                                    'Washer',
                                    'Fire extinguisher',
                                    'Hot water',
                                    'Lock on bedroom door',
                                    'Dryer',
                                    'First aid kit',
                                    'Dishes and silverware',
                                    'Oven',
                                    'Refrigerator',
                                    'Laptop friendly workspace']
                                           
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['amenities'] = X['amenities'].apply(lambda x: replace_brackets(x))
        X[['amenities']] = X[['amenities']].applymap(yaml.safe_load)
        
        for amenity in self.selected_amenitites:
            X[f'has_{amenity}'] = X.apply(lambda x: 1 if amenity in x['amenities'] else 0, axis=1)
        
        return X


# In[4]:


class CancellationPolicyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cancellation_type = pd.CategoricalDtype(categories=['flexible', 'moderate', 'strict'], ordered=True)
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['cancellation_policy'] = X['cancellation_policy'].apply(lambda x: 'strict' if 'strict' in x else x)
        X['cancellation_policy'] = X['cancellation_policy'].astype(self.cancellation_type).cat.codes
        return X


# In[6]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_new = X.drop(columns=self.columns_to_remove)
        return X_new


# In[7]:


class GroupImputers(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y = None):
        self.columns_means = {column: round(X.groupby(['room_type', 'neighbourhood_group_cleansed'])[column].mean()) 
                         for column in self.columns }
        
        return self
        
    def transform(self, X, y = None):
        for i, row in X.iterrows():
            for column in self.columns:
                if pd.isna(X.loc[i, column]):
                    room_type = row['room_type']
                    neighbourhood_group_cleansed = row['neighbourhood_group_cleansed']
                    value_to_input = self.columns_means[column][room_type, neighbourhood_group_cleansed]
                    X.loc[i, column] = value_to_input
        return X
        


# In[8]:


class HostAlwaysRespondsCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['host_always_responds'] = X.apply(lambda x: 1 if x['host_response_rate'] == '100%' else 0, axis=1)
        return X


# In[9]:


class HostIdentityVerifiedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for i, row in X.iterrows():
            if pd.isna(X.loc[i, 'host_identity_verified']):
                if X.loc[i, 'host_verifications'] in ['[]', 'None']:
                    X.loc[i, 'host_identity_verified'] = 'f'
                else:
                    X.loc[i, 'host_identity_verified'] = 't'
        return X


# In[10]:


class HostSinceImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['host_since'] = pd.to_datetime(X['host_since'], format='%Y-%m-%d')
        
        X_temp = X[['host_id', 'host_since']].copy()
        X_temp = X_temp.sort_values(by=['host_id'])
        X_temp.reset_index(inplace=True)
        X_temp['host_since'].fillna(method='bfill', inplace=True)
        X_temp = X_temp.sort_values(by='index')
        X_temp = X_temp.set_index('index')
        
        X['host_since'] = X_temp['host_since']
        return X


# In[11]:


class HostVerificationsImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for i, row in X.iterrows():
            if pd.isna(X.loc[i, 'host_verifications']):
                X.loc[i, 'host_verifications'] = 'None'
        return X


# In[12]:


class IsHostFastResponderCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fast_responses = ['within an hour', 'within a few hours', 'within a day']
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['is_host_fast_responder'] = X.apply(lambda x: 1 if x['host_response_time'] in self.fast_responses else 0, axis=1)
        return X


# In[13]:


class IsPhraseCreator(BaseEstimator, TransformerMixin):
    def __init__(self, column, phrase):
        self.column = column
        self.phrase = phrase
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X[f'is_{self.phrase}'] = X.apply(lambda x: 1 if x[self.column] == self.phrase else 0, axis=1)
        return X
    


# In[ ]:


class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.si = SimpleImputer(strategy='mean')
        self.si.fit(X)
        return self
    def transform(self, X, y=None):
        X_new = pd.DataFrame(columns=X.columns, data=self.si.transform(X))
        return X_new


# In[14]:


class NightsAvgCreator(BaseEstimator, TransformerMixin):
    def __init__(self, option):
        self.option = option
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.option == 'min':
            X['min_nights_avg'] = (X['minimum_nights'] + X['minimum_minimum_nights'] + X['maximum_minimum_nights'])/3
            X['min_nights_avg'] = X['min_nights_avg'].apply(lambda x: round(x, 1))
        elif self.option == 'max':
            X['max_nights_avg'] = (X['maximum_nights'] + X['minimum_maximum_nights'] + X['maximum_maximum_nights'])/3
            X['max_nights_avg'] = X['max_nights_avg'].apply(lambda x: round(x, 1))
        return X


# In[15]:


class ReviewColumnsImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['review_scores_location', 'review_scores_rating']):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X.apply(lambda x: 0.0 if x['number_of_reviews'] == 0 else x[column], axis=1)
        return X


# In[16]:


class ToNumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for col in self.columns:
            X[col] = X[col].apply(lambda x: remove_nonnumeric_chars(x))
            X[col] = X[col].astype('float')
        return X


# In[17]:


class TrueFalseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for col in self.columns:
            X[col] = X[col].apply(lambda x: 1 if x == 't' else 0)
        return X


# In[18]:


def remove_nonnumeric_chars(x):
    if not pd.isna(x):
        x = x.replace('$', '')
        x = x.replace(',', '')
    return x


# In[19]:


def replace_brackets(amenities):
    amenities = amenities.replace('{', '[')
    amenities = amenities.replace('}', ']')
    return amenities

