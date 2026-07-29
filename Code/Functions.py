#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import pickle
from plotly import graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.model_selection import KFold
import matplotlib.ticker as mticker


# ## Data 

# In[2]:


weekly_patterns_folder = 'Data_2018_2019'
naics_data_filepath = '2-6 digit_2017_Codes.csv'
census_df = pd.read_csv('census_new.csv')
poi_df = pd.read_csv('number_of_pois_visited_2018_2019.csv')
movement_data_folder = '2018_2019_movement_data'


# In[4]:


def create_dfs(weekly_patterns_folder):
    list_dfs = []
    for root, _, files in os.walk(weekly_patterns_folder):  
        for file in sorted(files):
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, dtype=str)
                df = df.rename(columns={'poi_cbg': 'poi_id', 'visitor_cbg': 'visitor_id'})
                df['naics_code'] = df['naics_code'].str[:-2]
                df['count'] = df['count'].astype(int)
                list_dfs.append(df)
    return list_dfs


def filter_dfs(df_list, column, starts_with):
    new_list = []
    for df in df_list:
        if starts_with == '':
            new_list.append(df)
        else:
            df = df[df[column].astype(str).str.startswith(starts_with)]
            new_list.append(df)
    return new_list
    
def convert_to_tracts(df_list):
    df_list_new = []
    for df in df_list:
        df_new = df.copy()
        df_new['poi_id'] = df_new['poi_id'].str[5:-1]
        df_new['visitor_id'] = df_new['visitor_id'].str[5:-1]
        df_new = df_new.groupby(['naics_code', 'poi_id', 'visitor_id'], as_index=False)['count'].sum()
        df_list_new.append(df_new)
    return df_list_new
    
def get_sparse_matrix(df, tracts): 
    num_tracts = len(tracts)
    directed_matrix = np.zeros((num_tracts, num_tracts),dtype=float)
    for index, row in df.iterrows():
        home_index = np.where(tracts == row['visitor_id'])
        poi_index = np.where(tracts == row['poi_id'])
        directed_matrix[home_index, poi_index] = row['count']
    sparse_matrix = csr_matrix(directed_matrix)
    return sparse_matrix
    
def get_tracts(df_list):
    df_combined = pd.concat(df_list)
    ids = np.unique(df_combined[['poi_id', 'visitor_id']].values)
    return ids
    
def build_matrices(df_list, tracts=None):
    mats = []
    if tracts is None:
        tracts = get_tracts(df_list)
    for df in df_list:
        mats.append(get_sparse_matrix(df, tracts))
    return mats
    
def get_total_movements(mat_list):
    movements = []
    for mat in mat_list:
        sums = np.sum(mat)
        movements.append(sums)
    movements = np.array(movements)
    return movements
    
def get_degrees_in(mat_list):
    degrees = []
    for mat in mat_list:
        sums = mat.sum(axis=0)
        sums = np.array(sums).flatten()
        degrees.append(sums)
    degrees = np.array(degrees)
    return degrees

def get_degrees_out(mat_list):
    degrees = []
    for mat in mat_list:
        sums = np.array(mat.sum(axis=1)).flatten()
        degrees.append(sums)
    degrees = np.array(degrees)
    return degrees
    
def get_naics_df(naics_data_filepath, num_digits=None, starts_with=None):
    df = pd.read_csv(naics_data_filepath, usecols=[1,2], skiprows=1, dtype=str)
    df.columns = ['Code', 'Name']
    if starts_with is not None:
        df = df[df['Code'].astype(str).str.startswith(str(starts_with))]
    else:
        df = df
    if num_digits is not None:
        df = df[df['Code'].astype(str).str.len() == num_digits]
    else:
        df = df
    return df
    
def merge_dfs(df1, df2, df3, omit_tracts):
    df1['Tract'] = df1['Tract'].astype(str)
    df2['Tract'] = df2['Tract'].astype(str)
    df3['Tract'] = df3['Tract'].astype(str)
    df = df1.merge(df2, on='Tract', how='outer').merge(df3, on='Tract', how='outer')
    df = df[~df['Tract'].isin(omit_tracts)]
    df = df.fillna(0)
    return df
    
def get_dates(weekly_patterns_folder):
    date_list = []
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")  
    for root, dirs, files in os.walk(weekly_patterns_folder):
        for file in files:
            match = date_pattern.search(file)
            if match:
                date_list.append(match.group())
    return sorted(set(date_list))  

def generate_movement_data(weekly_patterns_folder, movement_data_folder, naics_code=''):
    dfs = create_dfs(weekly_patterns_folder) 
    dfs = filter_dfs(dfs, 'visitor_id', '48201') 
    dfs = filter_dfs(dfs, 'naics_code', naics_code) 
    dfs = convert_to_tracts(dfs) 
    tracts = get_tracts(dfs)  
    mats = build_matrices(dfs, tracts) 
    deg_in = get_degrees_in(mats) 
    deg_out = get_degrees_out(mats) 
    total_movements = get_total_movements(mats)
    dict = (
        {'Tract':tracts,
        'In-Degree':deg_in,
        'Out-Degree':deg_out,
        'Total Movements':total_movements}
        )
    if naics_code == '':
        filename = f'{movement_data_folder}/all.pkl'
    else:
        filename = f'{movement_data_folder}/{naics_code}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(dict, f)
    return None

def load_movement_data(movement_data_folder, naics_code=''):
    if naics_code == '0' or naics_code == '' or naics_code == 'all' or naics_code == '00':
        filename = f'{movement_data_folder}/all.pkl'
    else:
        filename = f'{movement_data_folder}/{naics_code}.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            net_stats = pickle.load(f)
    else:
        print('No movement data for NAICS: ', naics_code)
        return None
    return net_stats

def get_degrees_mean(week_start, week_stop, movements, anomalous_week=None): # Time-averaged degree prediction
    df = pd.DataFrame({
        'Tract': movements['Tract'],
        'Mean In-Degree': movements['In-Degree'][week_start:week_stop+1,:].mean(axis=0),
        'Mean Out-Degree': movements['Out-Degree'][week_start:week_stop+1,:].mean(axis=0)
    })
    
    if anomalous_week is not None and anomalous_week != []:
        df['Anomalous In-Degree'] = movements['In-Degree'][anomalous_week, :]
        df['Anomalous Out-Degree'] = movements['Out-Degree'][anomalous_week, :]      
    return df

def get_degrees(week_start, week_stop, movements, anomalous_week): # Weekly degree prediction
    records = [] 
    for w in range(week_start, week_stop + 1):
        week_df = pd.DataFrame({'Tract': movements['Tract'],
                                'Week': w,
            'In_Degree': movements['In-Degree'][w, :],
            'Out_Degree': movements['Out-Degree'][w, :]})
        if anomalous_week is not None and anomalous_week != []:
            anomalous_in  = movements['In-Degree'][anomalous_week, :]
            anomalous_out = movements['Out-Degree'][anomalous_week, :]
            week__df = pd.DataFrame({
            'Anomalous_In':  anomalous_in,
            'Anomalous_Out': anomalous_out})
        records.append(week_df)
    df = pd.concat(records, axis=0, ignore_index=True)
    return df

def plot_observed(df, target, ax, label='', color='',marker=''):
    y = df[target].values
    y = y[y > 0] 
    if len(y) == 0:
        return
    counts, bins = np.histogram(y, bins=15)
    bins = (bins[1:] + bins[:-1]) / 2  
    counts = counts / counts.sum()  
    ax.loglog(bins, counts, linestyle='',color=color, marker=marker, alpha=0.4, markersize=6,label=label)

def plot_observed_vs_predicted(model_function,df,features, target,ax,label='',color='',markersize=12, alpha=0.4, marker=''):
    y, y_pred, r2 = model_function(df,features,target,return_plot_data=True)
    ax.loglog(y,y_pred,linestyle='',color=color,alpha=alpha,label=label,marker=marker )
    return None


# In[ ]:




