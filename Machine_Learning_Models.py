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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


# ## Multivariate Linear Regression (MLR)

# In[ ]:


def run_regression(df, features, target, return_plot_data=False):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=43)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    kf = KFold(n_splits=5, random_state=43, shuffle=True)
    fold_r2_train = []
    fold_r2_val = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        lr = LinearRegression()
        lr.fit(X_tr, y_tr)
        y_pred = lr.predict(X_test_scaled)
        r2_list.append(r2_score(y_test, y_pred))
    final_model = LinearRegression()
    final_model.fit(X_train_scaled, y_train)
    final_y_pred = final_model.predict(X_test_scaled)
    final_r2 = r2_score(y_test, final_y_pred)
    if return_plot_data:
        return y_test, final_y_pred, final_r2
    coefs =  [final_r2]
    names = ['R² (Model Fit)']
    return pd.DataFrame({'Name': names, 'Value': coefs})


# ## ElasticNet Regression (ENR)

# In[5]:


def run_elasticnet(df, features, target, n_splits_outer, random_state):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0], 
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 1.0] }
    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    fold_r2_train = []
    fold_r2_val = []
    best_params_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        model=ElasticNet(random_state=random_state)
        grid = GridSearchCV(model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
    best_global_params = (pd.DataFrame(best_params_per_fold).mode().iloc[0].to_dict())
    final_model =ElasticNet(**best_global_params)
    final_model.fit(X_train_scaled, y_train)
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    summary_df = pd.DataFrame({'Metric': ['Train CV R²', 'Val CV R²', 'Test R²'],
        'Mean': [ np.mean(fold_r2_train),np.mean(fold_r2_val),test_r2],
        'Std': [ np.std(fold_r2_train), np.std(fold_r2_val),  0.0 ]})
    return {'summary_df': summary_df, 'best_hyperparameters': best_global_params}


# ## Ridge Regression (RR)

# In[6]:


def run_ridge(df, features, target, n_splits_outer, random_state):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 500]}
    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    fold_r2_train = []
    fold_r2_val = []
    best_params_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        model=  Ridge()
        grid = GridSearchCV(model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
    best_global_params = (pd.DataFrame(best_params_per_fold).mode().iloc[0].to_dict())
    final_model = Ridge(**best_global_params)
    final_model.fit(X_train_scaled, y_train)
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    summary_df = pd.DataFrame({
        'Metric': ['Train CV R²', 'Val CV R²', 'Test R²'],
        'Mean': [ np.mean(fold_r2_train),np.mean(fold_r2_val),test_r2],
        'Std': [ np.std(fold_r2_train), np.std(fold_r2_val),  0.0 ]})
    return {'summary_df': summary_df, 'best_hyperparameters': best_global_params}


# ## Huber regression (HR)

# In[7]:


def run_huber(df, features, target, n_splits_outer, random_state):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {'epsilon': [1.1, 1.35, 1.5, 2.0],
    'alpha': [0.0001, 0.001, 0.01, 0.1], 
    'max_iter': [100, 200, 500]} 
    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    fold_r2_train = []
    fold_r2_val = []
    best_params_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        model=HuberRegressor()
        grid = GridSearchCV(model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
    best_global_params = (pd.DataFrame(best_params_per_fold).mode().iloc[0].to_dict())
    best_global_params['max_iter'] = int(best_global_params['max_iter'])
    final_model =HuberRegressor(**best_global_params)
    final_model.fit(X_train_scaled, y_train)
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    summary_df = pd.DataFrame({'Metric': ['Train CV R²', 'Val CV R²', 'Test R²'],
        'Mean': [ np.mean(fold_r2_train),np.mean(fold_r2_val),test_r2],
        'Std': [ np.std(fold_r2_train), np.std(fold_r2_val),  0.0 ]})
    return {'summary_df': summary_df, 'best_hyperparameters': best_global_params}


# ## Random Forest Regression (RFR)

# In[10]:


def run_random_forest(df, features, target, n_splits_outer=5, random_state=43, clip_extremes=True):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    y_log = np.log1p(y)    
    X_train, X_test, y_train_log, y_test_log = train_test_split( X, y_log, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {
         'n_estimators': [100, 200, 300, 400],
        'max_features': ['sqrt', 'log2', 0.5],
        'max_depth': [10, 20, None],
         'min_samples_leaf': [1, 2, 5, 10]}
    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    fold_r2_train, fold_r2_val, fold_importances, best_params_per_fold = [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train_log[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train_log[val_idx]
        rf = RandomForestRegressor(random_state=random_state + fold, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
        fold_importances.append(best_model.feature_importances_)
    fold_importances = np.array(fold_importances)
    best_idx = np.argmax(fold_r2_val)
    best_global_params = best_params_per_fold[best_idx]
    for param in ['max_depth', 'min_samples_leaf', 'n_estimators']:
        val = best_global_params.get(param)
        if val is not None:
           best_global_params[param] = int(val)
    final_model = RandomForestRegressor(**best_global_params, random_state=random_state, n_jobs=-1)#Train final model on full training set
    final_model.fit(X_train_scaled, y_train_log)
    y_test_pred_log = final_model.predict(X_test_scaled)  
    if clip_extremes: 
        upper = np.percentile(y_test_pred_log, 99)
        lower = np.percentile(y_test_pred_log, 1)
        y_test_pred_log = np.clip(y_test_pred_log, lower, upper)
    test_r2_log = r2_score(y_test_log, y_test_pred_log) 
    y_test_pred_raw = np.expm1(y_test_pred_log)  
    y_test_orig = np.expm1(y_test_log)
    test_r2_raw = r2_score(y_test_orig, y_test_pred_raw)
    feature_df = pd.DataFrame({
        "Feature": features,
        "Importance Mean (Train CV)": fold_importances.mean(axis=0),
        "Importance Std (Train CV)": fold_importances.std(axis=0),
        "Importance Final Model (Test)": final_model.feature_importances_
    }).sort_values("Importance Mean (Train CV)", ascending=False)
    summary_df = pd.DataFrame({
        'Metric': ['Train CV R² ', 'Val CV R² ', 'Final Test R² ', 'Final Test R² (raw)'],
        'Mean': [np.mean(fold_r2_train), np.mean(fold_r2_val), test_r2_log, test_r2_raw],
        'Std': [np.std(fold_r2_train), np.std(fold_r2_val), 0.0, 0.0]})
    return {'summary_df': summary_df,'best_hyperparameters': best_global_params}


# ## XGBoost Regression (XGBR)

# In[9]:


def run_xgboost(df, features, target, n_splits_outer, random_state):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {
        'max_depth': [1, 2, 3],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [5, 10, 20],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1, 5, 10]}
    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    fold_r2_train = []
    fold_r2_val = []
    best_params_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        base_model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=500,random_state=random_state)
        grid = GridSearchCV(base_model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
    best_global_params = (pd.DataFrame(best_params_per_fold).mode().iloc[0].to_dict())
    best_global_params['max_depth'] = int(best_global_params['max_depth'])
    final_model = xgb.XGBRegressor(**best_global_params,objective='reg:squarederror',n_estimators=500,random_state=random_state)
    final_model.fit(X_train_scaled, y_train)
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    summary_df = pd.DataFrame({
        'Metric': ['Train CV R²', 'Val CV R²', 'Test R²'],
        'Mean': [np.mean(fold_r2_train),np.mean(fold_r2_val),test_r2],
        'Std': [np.std(fold_r2_train),np.std(fold_r2_val),0.0] })
    return {'summary_df': summary_df,'best_hyperparameters': best_global_params}


# ##  Extra Trees Regression (ETR)

# In[8]:


def run_extra_trees(df, features, target, n_splits_outer=5, random_state=43):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [20, 40],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 5],
        'max_features': ['sqrt', 0.7]}
    kf = KFold(n_splits=n_splits_outer,shuffle=True,random_state=random_state)
    fold_r2_train = []
    fold_r2_val = []
    best_params_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        model= ExtraTreesRegressor()
        grid = GridSearchCV(model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
    best_global_params = (pd.DataFrame(best_params_per_fold).mode().iloc[0].to_dict())
    best_global_params['n_estimators'] = int(best_global_params['n_estimators'])
    best_global_params['max_depth'] = int(best_global_params['max_depth'])
    best_global_params['min_samples_leaf']=int(best_global_params['min_samples_leaf'])
    best_global_params['min_samples_split'] = int(best_global_params['min_samples_split'])
    final_model = ExtraTreesRegressor(**best_global_params)
    final_model.fit(X_train_scaled, y_train)
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    summary_df = pd.DataFrame({
        'Metric': ['Train CV R²', 'Val CV R²', 'Test R²'],
        'Mean': [ np.mean(fold_r2_train),np.mean(fold_r2_val),test_r2],
        'Std': [ np.std(fold_r2_train), np.std(fold_r2_val),  0.0 ]})
    return {'summary_df': summary_df, 'best_hyperparameters': best_global_params}
        


# ## Nearest Neighbor Regression (NNR)

# In[11]:


def run_neighborregressor(df, features, target, n_splits_outer, random_state):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']}
    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    fold_r2_train = []
    fold_r2_val = []
    best_params_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        model=KNeighborsRegressor()
        grid = GridSearchCV(model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
    best_global_params = (pd.DataFrame(best_params_per_fold).mode().iloc[0].to_dict())
    final_model = KNeighborsRegressor(**best_global_params)
    final_model.fit(X_train_scaled, y_train)
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    summary_df = pd.DataFrame({'Metric': ['Train CV R²', 'Val CV R²', 'Test R²'],
        'Mean': [ np.mean(fold_r2_train),np.mean(fold_r2_val),test_r2],
        'Std': [ np.std(fold_r2_train), np.std(fold_r2_val),  0.0 ]})
    return {'summary_df': summary_df, 'best_hyperparameters': best_global_params}


# ## Gradident Boosting Regression (GBR)

# In[12]:


def run_gradient_boosting(df, features, target, n_splits_outer, random_state):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {'n_estimators': [300, 500],
    'learning_rate': [0.01, 0.05],
    'max_depth': [1, 2],
    'min_samples_split': [20, 50],
    'min_samples_leaf': [10, 20, 50],
    'subsample': [0.6, 0.8],
    'max_features': ['sqrt', 0.5]}
    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    fold_r2_train = []
    fold_r2_val = []
    best_params_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        model=GradientBoostingRegressor()
        grid = GridSearchCV(model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
    best_global_params = (pd.DataFrame(best_params_per_fold).mode().iloc[0].to_dict())
    best_global_params['n_estimators'] = int(best_global_params['n_estimators'])
    best_global_params['max_depth'] = int(best_global_params['max_depth'])
    best_global_params['min_samples_split'] = int(best_global_params['min_samples_split'])
    best_global_params['min_samples_leaf']= int(best_global_params['min_samples_leaf'])
    final_model=GradientBoostingRegressor(**best_global_params)
    final_model.fit(X_train_scaled, y_train)
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    summary_df = pd.DataFrame({'Metric': ['Train CV R²', 'Val CV R²', 'Test R²'],
        'Mean': [ np.mean(fold_r2_train),np.mean(fold_r2_val),test_r2],
        'Std': [ np.std(fold_r2_train), np.std(fold_r2_val),  0.0 ]})
    return {'summary_df': summary_df, 'best_hyperparameters': best_global_params}


# ## Multilayer Perceptron Regression (MLPR)

# In[13]:


def run_mlp(df, features, target, n_splits_outer, random_state):
    df_clean = df.dropna(subset=features + [target])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target])
    X = df_clean[features].values
    y = df_clean[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {'max_iter': [100,200,500,1000],
               'hidden_layer_sizes': [(50,), (100,), (50,50)],
        'learning_rate_init': [0.001, 0.01,0.1,0.5]  }
    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    fold_r2_train = []
    fold_r2_val = []
    best_params_per_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]
        model=MLPRegressor(tol=0.1,early_stopping=True)
        grid = GridSearchCV(model,param_grid=param_grid,cv=3,scoring='r2',n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)
        y_tr_pred = best_model.predict(X_tr)
        y_val_pred = best_model.predict(X_val)
        fold_r2_train.append(r2_score(y_tr, y_tr_pred))
        fold_r2_val.append(r2_score(y_val, y_val_pred))
    best_global_params = (pd.DataFrame(best_params_per_fold).mode().iloc[0].to_dict())
    best_global_params['max_iter'] = int(best_global_params['max_iter'])
    final_model=MLPRegressor(**best_global_params,tol=0.1,early_stopping=True)
    final_model.fit(X_train_scaled, y_train)
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    summary_df = pd.DataFrame({'Metric': ['Train CV R²', 'Val CV R²', 'Test R²'],
        'Mean': [ np.mean(fold_r2_train),np.mean(fold_r2_val),test_r2],
        'Std': [ np.std(fold_r2_train), np.std(fold_r2_val),  0.0 ]})
    return {'summary_df': summary_df, 'best_hyperparameters': best_global_params}


# In[ ]:




