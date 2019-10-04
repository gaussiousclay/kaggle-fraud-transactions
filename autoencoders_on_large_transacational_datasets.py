# -*- coding: utf-8 -*-
"""Autoencoders on Large Transacational Datasets
    created by:sailesh mohanty
    created on: August 12,2019
"""

#@title Install PyDrive Requirements
!pip install -U -q PyDrive

from google.colab import drive
drive.mount("/content/drive")

#@title Importing Packages
# importing utilities
import os
import sys
from datetime import datetime
import io
import urllib
import gc


# importing data science libraries
import pandas as pd
import random as rd
import numpy as np

# importing pytorch libraries
import torch
from torch import nn
from torch import autograd
from torch.utils.data import DataLoader

import tensorflow as tf

# import visualization libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from IPython.display import Image, display
sns.set_style('darkgrid')

# import colab libraries
from google.colab import files
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# ignore potential warnings
import warnings
warnings.filterwarnings("ignore")

#@title Check Architectural Requirements
# print CUDNN backend version
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The CUDNN backend version: {}'.format(now, torch.backends.cudnn.version()))


!nvidia-smi



USE_CUDA = False


# print current PyTorch version
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The PyTorch version: {}'.format(now, torch.__version__))


# print current Python version
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The Python version: {}'.format(now, sys.version))

#@title Connect to Google Drive
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

#@title Read and Preprocess File
def read_file(link,filename):
  fluff, id = link.split('=')
  print (id)
  downloaded = drive.CreateFile({'id':id}) 
  downloaded.GetContentFile(filename)
  file = pd.read_csv(filename)
  return file


def get_mail_provider_detail(x):
    if(type(x)==float):
        split_str = ['']
    else:
        split_str = x.split('.')
        
    
    if(len(split_str)==3):
        split_str[1] = split_str[1] +'.'+split_str[2]
        del split_str[-1]    
    company = ''
    industry = ''

    if(split_str[0] == 'gmail'):
        company = 'Google'
        industry = 'Technology'
    elif(split_str[0] in ['outlook','live','msn','passport','hotmail']):
        company = 'Microsoft'
        industry = 'Technology'
    elif(split_str[0] in ['yahoo','ymail','rocketmail']):
        company = 'Yahoo'
        industry = 'Technology'
    elif(split_str[0] in ['verizon','aol','aim']):
        company = 'Verizon'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['mail','gmx']):
        company = 'United Internet'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['icloud','me','mac']):
        company = 'Apple'
        industry = 'Technology'
    elif(split_str[0] in ['comcast']):
        company = 'Xfinity_Comcast'
        industry = 'Streaming'
    elif(split_str[0] in ['sbcglobal','att','bellsouth','prodigy']):
        company = 'AT_T'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['cox']):
        company = 'Cox Communications'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['web']):
        company = 'WEB_DE'
        industry = 'Internet'
    elif(split_str[0] in ['optonline','suddenlink']):
        company = 'Altice'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['charter']):
        company = 'Charter Communications'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['earthlink','windstream']):
        company = 'Windstream'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['embarqmail','centurylink','q']):
        company = 'Century Link'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['juno','netzero']):
        company = 'United Online'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['frontier','frontiernet']):
        company = 'Frontier Communications'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['roadrunner','twc','cfl','sc']):
        company = 'Time Warner'
        industry = 'Streaming'
    elif(split_str[0] in ['protonmail']):
        company = 'Proton'
        industry = 'Mail Services'
    elif(split_str[0] in ['servicios-ta']):
        company = 'Airport Services'
        industry = 'Transportation'  
    elif(split_str[0] in ['scranton']):
        company = 'Scranton University'
        industry = 'Education'
    elif(split_str[0] in ['ptd']):
        company = 'Service Electric'
        industry = 'Utilities'
    elif(split_str[0] in ['cableone']):
        company = 'Cable One Communications'
        industry = 'Broadband and Telecommunications'
    elif(split_str[0] in ['anonymous']):
        company = 'Anonymous'
        industry = 'Unknown'
    else:
        company = 'Unavailable'
        industry = 'Unavailable'

    return ([company,industry])

  
def get_mail_country_details(x):
    if(type(x)==float):
        split_str = ['']
    else:
        split_str = x.split('.')
        
    
    if(len(split_str)==3):
        split_str[1] = split_str[1] +'.'+split_str[2]
        del split_str[-1]    
    
    country=''
        
    y = get_mail_provider_detail(x)
    company = y[0]

    if(split_str[0]=='' or len(split_str)==1):
        country = 'Unavailable'
    elif(split_str[1] == 'de'):
        country = 'Germany'
    elif(split_str[1] == 'co.uk'):
        country = 'Great Britain'
    elif(split_str[1] == 'co.jp'):
        country = 'Japan'
    elif(split_str[1] == 'fr'):
        country = 'France'
    elif(split_str[1] == 'es'):
        country = 'Spain'
    elif(split_str[1] in ['com.mx','mx','net.mx']):
        country = 'Mexico'
    elif(split_str[1] == 'com' and company in ['Google','Microsoft','Yahoo','Apple']):
        country = 'International'
    elif(split_str[1] == 'anonymous'):
        country = 'Anonymous'    
    else:
        country = 'US'
    
    return(country)

#@title Read Datasets and optimize for memory
traini = read_file("https://drive.google.com/open?id=1sQT471Q6vxKAmPEjgtFMLUcqU7UoCT0L","train_identity.csv")
traintr = read_file("https://drive.google.com/open?id=1Y_ycrplGLggEDmeEq9Y_YSe-MymwSuca","train_transaction.csv")

testi = read_file("https://drive.google.com/open?id=1TRP_59SRKFtZbp5mH2j2GEwTfZrOGaQC","test_identity.csv")
testtr = read_file("https://drive.google.com/open?id=1TTK8Mvj1qxewnDpaQ7E_Smvdzwm62sdk","test_transaction.csv")

train = pd.merge(traintr, traini, on='TransactionID', how='left')
test = pd.merge(testtr, testi, on='TransactionID', how='left')

print(f'Train dataset: {train.shape[0]} rows & {train.shape[1]} columns')
print(f'Test dataset: {test.shape[0]} rows & {test.shape[1]} columns')

print('Applying Memory Reduction Techniques')

def reduce_mem_usage(df, verbose=True):
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  start_mem = df.memory_usage().sum() / 1024**2    
  for col in df.columns:
    col_type = df[col].dtypes
    if col_type in numerics:
      c_min = df[col].min()
      c_max = df[col].max()
      if str(col_type)[:3] == 'int':
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
          df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
          df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
          df[col] = df[col].astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
          df[col] = df[col].astype(np.int64)  
      else:
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
          df[col] = df[col].astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
          df[col] = df[col].astype(np.float32)
        else:
          df[col] = df[col].astype(np.float64)    
  end_mem = df.memory_usage().sum() / 1024**2
  if verbose: 
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
  return df


#@title Decoding column names in Train and Test
train['Mail_Details'] = train['P_emaildomain'].apply(lambda x: get_mail_provider_detail(x))
train['P_Company'] = train['Mail_Details'].apply(lambda x: x[0])
train['P_Industry'] = train['Mail_Details'].apply(lambda x: x[1])
train['P_Country'] = train['P_emaildomain'].apply(lambda x:get_mail_country_details(x))
train['Mail_Details'] = train['R_emaildomain'].apply(lambda x: get_mail_provider_detail(x))
train['R_Company'] = train['Mail_Details'].apply(lambda x: x[0])
train['R_Industry'] = train['Mail_Details'].apply(lambda x: x[1])
train['R_Country'] = train['R_emaildomain'].apply(lambda x:get_mail_country_details(x))
train.drop(['Mail_Details','P_emaildomain','R_emaildomain'],axis=1,inplace=True)
train['nulls1'] = train.isna().sum(axis=1)

test['Mail_Details'] = test['P_emaildomain'].apply(lambda x: get_mail_provider_detail(x))
test['P_Company'] = test['Mail_Details'].apply(lambda x: x[0])
test['P_Industry'] = test['Mail_Details'].apply(lambda x: x[1])
test['P_Country'] = test['P_emaildomain'].apply(lambda x:get_mail_country_details(x))
test['Mail_Details'] = test['R_emaildomain'].apply(lambda x: get_mail_provider_detail(x))
test['R_Company'] = test['Mail_Details'].apply(lambda x: x[0])
test['R_Industry'] = test['Mail_Details'].apply(lambda x: x[1])
test['R_Country'] = test['R_emaildomain'].apply(lambda x:get_mail_country_details(x))
test.drop(['Mail_Details','P_emaildomain','R_emaildomain'],axis=1,inplace=True)
test['nulls1'] = test.isna().sum(axis=1)

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

del testi, testtr, traini, traintr
gc.collect()

#@title Remove Garbage Columns
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))
cols_to_drop.remove('isFraud')
print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

print(f'Train dataset after removing garbage: {train.shape[0]} rows & {train.shape[1]} columns')
print(f'Test dataset after removing garbage: {test.shape[0]} rows & {test.shape[1]} columns')

#@title Feature Engineering
def addNewFeatures(data):        
  data['TransactionAmt_to_mean_card1'] = data['TransactionAmt'] / data.groupby(['card1'])['TransactionAmt'].transform('mean')
  data['TransactionAmt_to_mean_card4'] = data['TransactionAmt'] / data.groupby(['card4'])['TransactionAmt'].transform('mean')
  data['TransactionAmt_to_std_card1'] = data['TransactionAmt'] / data.groupby(['card1'])['TransactionAmt'].transform('std')
  data['TransactionAmt_to_std_card4'] = data['TransactionAmt'] / data.groupby(['card4'])['TransactionAmt'].transform('std')

  data['id_02_to_mean_card1'] = data['id_02'] / data.groupby(['card1'])['id_02'].transform('mean')
  data['id_02_to_mean_card4'] = data['id_02'] / data.groupby(['card4'])['id_02'].transform('mean')
  data['id_02_to_std_card1'] = data['id_02'] / data.groupby(['card1'])['id_02'].transform('std')
  data['id_02_to_std_card4'] = data['id_02'] / data.groupby(['card4'])['id_02'].transform('std')

  data['D15_to_mean_card1'] = data['D15'] / data.groupby(['card1'])['D15'].transform('mean')
  data['D15_to_mean_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('mean')
  data['D15_to_std_card1'] = data['D15'] / data.groupby(['card1'])['D15'].transform('std')
  data['D15_to_std_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('std')

  data['D15_to_mean_addr1'] = data['D15'] / data.groupby(['addr1'])['D15'].transform('mean')
  data['D15_to_mean_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('mean')
  data['D15_to_std_addr1'] = data['D15'] / data.groupby(['addr1'])['D15'].transform('std')
  data['D15_to_std_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('std')

  data['uid'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)

  data['uid2'] = data['uid'].astype(str)+'_'+data['card3'].astype(str)+'_'+data['card5'].astype(str)

  data['uid3'] = data['uid2'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)

  data['TransactionAmt'] = np.log1p(data['TransactionAmt'])

  data['D9'] = np.where(data['D9'].isna(),0,1)
    
  return data

train = addNewFeatures(train)
test = addNewFeatures(test)
train = train.replace(np.inf,999)
test = test.replace(np.inf,999)


train["latest_browser"] = np.zeros(train.shape[0])
test["latest_browser"] = np.zeros(test.shape[0])

def setBrowser(df):
  df.loc[df["id_31"]=="samsung browser 7.0",'latest_browser']=1
  df.loc[df["id_31"]=="opera 53.0",'latest_browser']=1
  df.loc[df["id_31"]=="mobile safari 10.0",'latest_browser']=1
  df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
  df.loc[df["id_31"]=="firefox 60.0",'latest_browser']=1
  df.loc[df["id_31"]=="edge 17.0",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 69.0",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 67.0 for android",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 63.0 for android",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 63.0 for ios",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 64.0",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 64.0 for android",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 64.0 for ios",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 65.0",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 65.0 for android",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 65.0 for ios",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 66.0",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 66.0 for android",'latest_browser']=1
  df.loc[df["id_31"]=="chrome 66.0 for ios",'latest_browser']=1
  return df

train=setBrowser(train)
test=setBrowser(test)

print(f'Train dataset after feature engineering: {train.shape[0]} rows & {train.shape[1]} columns')
print(f'Test dataset after feature engineering: {test.shape[0]} rows & {test.shape[1]} columns')

#@title Encode Categorical Variables, define imputation and scaling methods
class ModifiedLabelEncoder(LabelEncoder):
  def fit_transform(self, y, *args, **kwargs):
    return super().fit_transform(y).reshape(-1, 1)

  def transform(self, y, *args, **kwargs):
    return super().transform(y).reshape(-1, 1)
  

class DataFrameSelector(BaseEstimator, TransformerMixin):
  def __init__(self, attr):
    self.attributes = attr
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    return X[self.attributes].values
  
  
noisy_cols = [
    'TransactionID','TransactionDT',                         
    'uid','uid2','uid3',                                    
    'id_30', 'id_33',
]
    

def encodeCategorical(df_train, df_test):
  for f in df_train.drop('isFraud', axis=1).columns:
    if df_train[f].dtype=='object' or df_test[f].dtype=='object': 
      lbl = preprocessing.LabelEncoder()
      lbl.fit(list(df_train[f].values) + list(df_test[f].values))
      df_train[f] = lbl.transform(list(df_train[f].values))
      df_test[f] = lbl.transform(list(df_test[f].values))
  return df_train, df_test


#@title Preparing for Training
# select categorical attributes to be "one-hot" encoded
y_train = train['isFraud']
train, test = encodeCategorical(train, test)
gc.collect()
trainid = train.pop('TransactionID')
isFraud = train.pop('isFraud')
train.drop(['TransactionDT','uid','uid2','uid3','id_30', 'id_33'],axis=1,inplace=True)
testid = test.pop('TransactionID')
test.drop(['TransactionDT','uid','uid2','uid3','id_30', 'id_33'],axis=1,inplace=True)
imp_mean = SimpleImputer()
scaler = preprocessing.MinMaxScaler()
train_transformed = pd.DataFrame(imp_mean.fit_transform(train),columns=train.columns)
train_transformed = pd.DataFrame(scaler.fit_transform(train_transformed),columns=train.columns)
test_transformed = pd.DataFrame(imp_mean.fit_transform(test),columns=test.columns)
test_transformed = pd.DataFrame(scaler.fit_transform(test_transformed),columns=test.columns)
del train,test
gc.collect()

print(f'Train dataset after pipeline execution: {train_transformed.shape[0]} rows & {train_transformed.shape[1]} columns')
print(f'Test dataset after pipeline execution: {test_transformed.shape[0]} rows & {test_transformed.shape[1]} columns')

#@title Preparing for Training
train_transformed['isFraud'] = isFraud
train_transformed['isFraud'].value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train_transformed, test_size=0.2, random_state=412)

X_train = X_train[X_train.isFraud == 0]
X_train = X_train.drop(['isFraud'], axis=1)
y_test = X_test['isFraud']
X_test = X_test.drop(['isFraud'], axis=1)

del train_transformed

#@title Enocder Network Architecture Definition
# implementation of the encoder network
class encoder(nn.Module):
  def __init__(self):    
    super(encoder, self).__init__()

    # specify layer 1 - in 368, out 256
    self.encoder_L1 = nn.Linear(in_features=X_train.shape[1], out_features=256, bias=True) # add linearity 
    nn.init.xavier_uniform_(self.encoder_L1.weight) # init weights according to [9]
    self.encoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) # add non-linearity according to [10]

    # specify layer 2 - in 256, out 128
    self.encoder_L2 = nn.Linear(256, 128, bias=True)
    nn.init.xavier_uniform_(self.encoder_L2.weight)
    self.encoder_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

    # specify layer 3 - in 128, out 64
    self.encoder_L3 = nn.Linear(128, 64, bias=True)
    nn.init.xavier_uniform_(self.encoder_L3.weight)
    self.encoder_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

    # specify layer 4 - in 64, out 32
    self.encoder_L4 = nn.Linear(64, 32, bias=True)
    nn.init.xavier_uniform_(self.encoder_L4.weight)
    self.encoder_R4 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

    # specify layer 5 - in 32, out 16
    self.encoder_L5 = nn.Linear(32, 16, bias=True)
    nn.init.xavier_uniform_(self.encoder_L5.weight)
    self.encoder_R5 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

    # specify layer 6 - in 16, out 8
    self.encoder_L6 = nn.Linear(16, 8, bias=True)
    nn.init.xavier_uniform_(self.encoder_L6.weight)
    self.encoder_R6 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

    # specify layer 7 - in 8, out 4
    self.encoder_L7 = nn.Linear(8, 4, bias=True)
    nn.init.xavier_uniform_(self.encoder_L7.weight)
    self.encoder_R7 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

    # specify layer 8 - in 4, out 3
    self.encoder_L8 = nn.Linear(4, 3, bias=True)
    nn.init.xavier_uniform_(self.encoder_L8.weight)
    self.encoder_R8 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

    # init dropout layer with probability p
    self.dropout = nn.Dropout(p=0.0, inplace=True)
        
  def forward(self, x):
    # define forward pass through the network
    x = self.encoder_R1(self.dropout(self.encoder_L1(x)))
    x = self.encoder_R2(self.dropout(self.encoder_L2(x)))
    x = self.encoder_R3(self.dropout(self.encoder_L3(x)))
    x = self.encoder_R4(self.dropout(self.encoder_L4(x)))
    x = self.encoder_R5(self.dropout(self.encoder_L5(x)))
    x = self.encoder_R6(self.dropout(self.encoder_L6(x)))
    x = self.encoder_R7(self.dropout(self.encoder_L7(x)))
    x = self.encoder_R8(self.encoder_L8(x))         # don't apply dropout to the AE bottleneck
    return x

#@title Encoder Network Initialization
# init training network classes / architectures
encoder_train = encoder()

# push to cuda if cudnn is available
if (torch.backends.cudnn.version() != None and USE_CUDA == True):
  encoder_train = encoder().cuda()
    
    
# print the initialized architectures
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] encoder architecture:\n\n{}\n'.format(now, encoder_train))

#@title Decoder Network Definition
class decoder(nn.Module):
  
  def __init__(self):
    super(decoder, self).__init__()
    # specify layer 1 - in 2, out 4
    self.decoder_L1 = nn.Linear(in_features=3, out_features=4, bias=True) # add linearity 
    nn.init.xavier_uniform_(self.decoder_L1.weight)  # init weights according to [9]
    self.decoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) # add non-linearity according to [10]
    # specify layer 2 - in 4, out 8
    self.decoder_L2 = nn.Linear(4, 8, bias=True)
    nn.init.xavier_uniform_(self.decoder_L2.weight)
    self.decoder_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
    # specify layer 3 - in 8, out 16
    self.decoder_L3 = nn.Linear(8, 16, bias=True)
    nn.init.xavier_uniform_(self.decoder_L3.weight)
    self.decoder_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
    # specify layer 4 - in 16, out 32
    self.decoder_L4 = nn.Linear(16, 32, bias=True)
    nn.init.xavier_uniform_(self.decoder_L4.weight)
    self.decoder_R4 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
    # specify layer 5 - in 32, out 64
    self.decoder_L5 = nn.Linear(32, 64, bias=True)
    nn.init.xavier_uniform_(self.decoder_L5.weight)
    self.decoder_R5 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
    # specify layer 6 - in 64, out 128
    self.decoder_L6 = nn.Linear(64, 128, bias=True)
    nn.init.xavier_uniform_(self.decoder_L6.weight)
    self.decoder_R6 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
    # specify layer 7 - in 128, out 256
    self.decoder_L7 = nn.Linear(128, 256, bias=True)
    nn.init.xavier_uniform_(self.decoder_L7.weight)
    self.decoder_R7 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
    # specify layer 8 - in 256, out 368
    self.decoder_L8 = nn.Linear(256, out_features = X_train.shape[1], bias=True)
    nn.init.xavier_uniform_(self.decoder_L8.weight)
    self.decoder_R8 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
    # init dropout layer with probability p
    self.dropout = nn.Dropout(p=0.0, inplace=True)
    
  def forward(self, x):
    # define forward pass through the network
    x = self.decoder_R1(self.dropout(self.decoder_L1(x)))
    x = self.decoder_R2(self.dropout(self.decoder_L2(x)))
    x = self.decoder_R3(self.dropout(self.decoder_L3(x)))
    x = self.decoder_R4(self.dropout(self.decoder_L4(x)))
    x = self.decoder_R5(self.dropout(self.decoder_L5(x)))
    x = self.decoder_R6(self.dropout(self.decoder_L6(x)))
    x = self.decoder_R7(self.dropout(self.decoder_L7(x)))
    x = self.decoder_R8(self.decoder_L8(x)) # don't apply dropout to the AE output
    return x

#@title Decoder Network Initialization
# init training network classes / architectures
decoder_train = decoder()

# push to cuda if cudnn is available
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
  decoder_train = decoder().cuda()
    
# print the initialized architectures
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] decoder architecture:\n\n{}\n'.format(now, decoder_train))

#@title Initialize Training Parameters
# define the optimization criterion / loss function
loss_function = nn.BCEWithLogitsLoss(reduction='mean')

# define learning rate and optimization strategy
learning_rate = 1e-3
encoder_optimizer = torch.optim.Adam(encoder_train.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder_train.parameters(), lr=learning_rate)

# specify training parameters
num_epochs = 50
mini_batch_size = 128


# convert pre-processed data to pytorch tensor
torch_dataset = torch.from_numpy(X_train.values).float()

# convert to pytorch tensor - none cuda enabled
dataloader = DataLoader(torch_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=0)
# note: we set num_workers to zero to retrieve deterministic results

# determine if CUDA is available at compute node
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
  dataloader = DataLoader(torch_dataset.cuda(), batch_size=mini_batch_size, shuffle=True)

#@title Start Training
# init collection of epoch losses
epoch_losses = []

# convert encoded transactional data to torch Variable
data = autograd.Variable(torch_dataset)

# train autoencoder model
for epoch in range(num_epochs):

    # init collection of epoch losses
    mini_batch_losses = []
    
    # init mini batch counter
    mini_batch_count = 0
    
    # determine if CUDA is available at compute node
    if(torch.backends.cudnn.version() != None) and (USE_CUDA == True):
        
        # set networks / models in GPU mode
        encoder_train.cuda()
        decoder_train.cuda()

    # set networks in training mode (apply dropout when needed)
    encoder_train.train()
    decoder_train.train()

    # start timer
    start_time = datetime.now()
        
    # iterate over all mini-batches
    for mini_batch_data in dataloader:

        # increase mini batch counter
        mini_batch_count += 1

        # convert mini batch to torch variable
        mini_batch_torch = autograd.Variable(mini_batch_data)

        # =================== (1) forward pass ===================================

        # run forward pass
        z_representation = encoder_train(mini_batch_torch) # encode mini-batch data
        mini_batch_reconstruction = decoder_train(z_representation) # decode mini-batch data
        
        # =================== (2) compute reconstruction loss ====================

        # determine reconstruction loss
        reconstruction_loss = loss_function(mini_batch_reconstruction, mini_batch_torch)
        
        # =================== (3) backward pass ==================================

        # reset graph gradients
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        # run backward pass
        reconstruction_loss.backward()
        
        # =================== (4) update model parameters ========================

        # update network parameters
        decoder_optimizer.step()
        encoder_optimizer.step()

        # =================== monitor training progress ==========================

        # print training progress each 1'000 mini-batches
        if mini_batch_count % 1000 == 0:
            
            # print the training mode: either on GPU or CPU
            mode = 'GPU' if (torch.backends.cudnn.version() != None) and (USE_CUDA == True) else 'CPU'
            
            # print mini batch reconstuction results
            now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
            end_time = datetime.now() - start_time
            print('[LOG {}] training status, epoch: [{:04}/{:04}], batch: {:04}, loss: {}, mode: {}, time required: {}'.format(now, (epoch+1), num_epochs, mini_batch_count, np.round(reconstruction_loss.item(), 4), mode, end_time))

            # reset timer
            start_time = datetime.now()
            
        # collect mini-batch loss
        mini_batch_losses.extend([np.round(reconstruction_loss.item(), 4)])

    # =================== evaluate model performance =============================
                                 
    # collect mean training epoch loss
    epoch_losses.extend([np.mean(mini_batch_losses)])
    
    # print training epoch results
    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    print('[LOG {}] training status, epoch: [{:04}/{:04}], loss: {:.10f}'.format(now, (epoch+1), num_epochs, np.mean(mini_batch_losses)))

    # =================== save model snapshot to disk ============================
    
    # save trained encoder model file to disk
    encoder_model_name = "ep_{}_encoder_model.pth".format((epoch+1))
    #uploaded = drive.CreateFile({'title':encoder_model_name})
    #uploaded.SetContentFile(encoder_train.state_dict())
    #uploaded.Upload()
    torch.save(encoder_train.state_dict(), encoder_model_name)

    # save trained decoder model file to disk
    decoder_model_name = "ep_{}_decoder_model.pth".format((epoch+1))
    #uploaded = drive.CreateFile({'title':decoder_model_name})
    #uploaded.SetContentFile(decoder_train.state_dict())
    #uploaded.Upload()
    torch.save(decoder_train.state_dict(), decoder_model_name)

# plot the training progress
plt.plot(range(0, len(epoch_losses)), epoch_losses)
plt.xlabel('[training epoch]')
plt.xlim([0, len(epoch_losses)])
plt.ylabel('[reconstruction-error]')
#plt.ylim([0.0, 1.0])
plt.title('AENN training performance')

# init training network classes / architectures
encoder_eval = encoder()
decoder_eval = decoder()

# load trained models
encoder_eval.load_state_dict(torch.load('/content/ep_50_encoder_model.pth'))
decoder_eval.load_state_dict(torch.load('/content/ep_50_decoder_model.pth'))

# convert encoded transactional data to torch Variable
torch_dataset_valid = torch.from_numpy(X_test.values).float()
data_valid = autograd.Variable(torch_dataset_valid)

# set networks in evaluation mode (don't apply dropout)
encoder_eval.eval()
decoder_eval.eval()

# reconstruct encoded transactional data
reconstruction = decoder_eval(encoder_eval(data_valid))

# init binary cross entropy errors
reconstruction_loss_transaction = np.zeros(reconstruction.size()[0])

# iterate over all detailed reconstructions
for i in range(0, reconstruction.size()[0]):

    # determine reconstruction loss - individual transactions
    reconstruction_loss_transaction[i] = loss_function(reconstruction[i], data_valid[i]).item()

    if(i % 100000 == 0):

        ### print conversion summary
        now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
        print('[LOG {}] collected individual reconstruction loss of: {:06}/{:06} transactions'.format(now, i, reconstruction.size()[0]))

y_test.value_counts()

# prepare plot
fig = plt.figure()
ax = fig.add_subplot(111)

# assign unique id to transactions
plot_data = np.column_stack((np.arange(len(reconstruction_loss_transaction)), reconstruction_loss_transaction))

# obtain regular transactions as well as global and local anomalies
regular_data = plot_data[y_test == 0]
fraud_data = plot_data[y_test == 1]

# plot reconstruction error scatter plot
ax.scatter(regular_data[:, 1], np.tanh(regular_data[:, 1]), c='C0', alpha=0.4, marker="o", label='regular') # plot regular transactions
ax.scatter(fraud_data[:, 1], np.tanh(fraud_data[:, 1]), c='C1', marker="^", label='fraud') # plot global outliers

# add plot legend of transaction classes
ax.legend(loc='best')

reconstruction_valid['RL_1'] = reconstruction_valid['Reconstruction_Loss']

