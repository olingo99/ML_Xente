import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE


def getDay(x):
    return float(''.join(x.split("T")[0].split("-")))

def getTime(x):
    time = x.split("T")[1].split(":")
    time[-1] = time[-1][:-1]
    return float(''.join(time))



class StringCleanTransformer(BaseEstimator, TransformerMixin):
    def fit (self, X,y = None):
        return self
    
    def transform(self, X, y=None):
        StringToClean = ["BatchId","AccountId","SubscriptionId","CustomerId", "ProviderId", "ProductId", "ChannelId", "ProductCategory"]
        for col in StringToClean:
            X[col] = X[col].apply(lambda x : x.split("_")[-1])
        return X


class DayTimeTransformer(BaseEstimator, TransformerMixin):
    def fit (self, X,y = None):
        return self
    
    def transform(self, X, y=None):
        X["TransactionStartDay"]  = X["TransactionStartTime"].apply(getDay)
        X["TransactionStartTime"] = X["TransactionStartTime"].apply(getTime)
        return X

class DropperTransformer(BaseEstimator, TransformerMixin):
    def fit (self, X,y = None):
        return self
    
    def transform(self, X, y=None):
        X.drop(['CurrencyCode', 'CountryCode', 'BatchId', 'CustomerId', 'SubscriptionId', 'ProductId', 'Amount'], axis=1, inplace=True)
        return X

class SignTransformer(BaseEstimator, TransformerMixin):
    def fit (self, X,y = None):
        return self
    
    def transform(self, X, y=None):
        X["Sign"] = X["Amount"].apply(lambda x : x>=0)
        return X

class OHTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit (self, X,y = None):
        return self
    
    def transform(self, X, y=None):
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        for elem in self.columns:
            OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[elem].values.reshape(-1,1)))
            OH_cols.rename(columns=lambda x: elem + str(x), inplace=True)
            OH_cols.index = X.index
            X = pd.concat([X, OH_cols], axis=1)
            X.drop(elem, axis=1, inplace=True)
        return X

# class SmoteTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, Y=None):
#         self.Y = columns

#     def fit (self, X,y = None):
#         return self
    
#     def transform(self, X, y=None):
#         SMOTE = SMOTE()
#         smote_X, smote_Y = SMOTE.fit_resample(X, self.Y)
#         return X