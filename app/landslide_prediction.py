#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
import sys
import warnings
warnings.filterwarnings('ignore')


# In[3]:


landslides = pd.read_csv('data\Global_Landslide_Catalog_Export.csv', sep=';')
landslides.head()


# In[4]:


# We will use a machine learning model to predict fatality counts for landslides. 
# The input features will be the landslide size, landslide trigger, admin division population, gazeteer distance to nearest city, localisation (longitude + lattitude)
# The output feature will be the fatality count

# We will first use a random forest regressor model

# create a new dataframe with the features and the output

landslides_df = landslides[['landslide_size', 'landslide_trigger', 'admin_division_population', 'gazeteer_distance', 'latitude', 'longitude', 'fatality_count']]
landslides_df.head()


# In[5]:


# print data types
landslides_df.dtypes


# In[6]:


# show missing values
landslides_df.isnull().sum()


# In[7]:


# drop rows with missing values
landslides_df = landslides_df.dropna()

# show missing values
landslides_df.isnull().sum()


# In[8]:


# convert admin division population to int (replace comma with nothing)
landslides_df['admin_division_population'] = landslides_df['admin_division_population'].str.replace(',', '')
landslides_df['admin_division_population'] = landslides_df['admin_division_population'].astype(int)

# same for fatality count
landslides_df['fatality_count'] = landslides_df['fatality_count'].str.replace(',', '')
landslides_df['fatality_count'] = landslides_df['fatality_count'].astype(int)

landslides_df.dtypes

print("shape: ", landslides_df.shape)


# In[9]:


# Convertir les variables catégorielles en variables one-hot
landslides_df = pd.get_dummies(landslides_df, columns=['landslide_size', 'landslide_trigger'])

# Afficher les premières lignes du nouveau dataframe
landslides_df.head()


# In[10]:


# normalize the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
landslides_df[['admin_division_population', 'gazeteer_distance', 'latitude', 'longitude']] = scaler.fit_transform(landslides_df[['admin_division_population', 'gazeteer_distance', 'latitude', 'longitude']])
landslides_df.head()


# In[11]:


landslides_df.dtypes


# In[12]:


# print correlation matrix
corr = landslides_df.corr()
corr.style.background_gradient(cmap='rocket')

# plot correlation heatmap
sns.heatmap(corr, cmap='rocket', annot=False)


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Séparer les features (X) et la variable cible (y)
X = landslides_df.drop('fatality_count', axis=1)
y = landslides_df['fatality_count']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de forêt aléatoire
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle sur les données d'entraînement
rf_model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = rf_model.predict(X_test)

# Évaluer les performances du modèle
mse_RF = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_RF}')


# In[14]:


# Afficher les 10 premières valeurs prédites et les valeurs réelles

for i in range(10):
    print(f'Predicted value: {y_pred[i]:.2f} - Actual value: {y_test.values[i]}')


# In[15]:


# plot feature importance
plt.figure(figsize=(20, 10))
sns.barplot(x=X.columns, y=rf_model.feature_importances_, palette="rocket")
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


# In[16]:


# plot hyperparameters of the model
from pprint import pprint

pprint(rf_model.get_params())


# In[17]:


# hyperparameter tuning with grid search

from sklearn.model_selection import GridSearchCV

# Définir les hyperparamètres à tester

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialiser le modèle de forêt aléatoire
rf_model = RandomForestRegressor(random_state=42)

# Initialiser la recherche sur grille avec les hyperparamètres à tester
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Entraîner le modèle sur les données d'entraînement
grid_search.fit(X_train, y_train)


# In[18]:


# Afficher les meilleurs hyperparamètres
print(grid_search.best_params_)

# Faire des prédictions sur les données de test
y_pred = grid_search.predict(X_test)

# Évaluer les performances du modèle
mse_RF_tuned = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_RF_tuned}')


# In[19]:


# Afficher les 10 premières valeurs prédites et les valeurs réelles
for i in range(10):
    print(f'Predicted value: {y_pred[i]:.2f} - Actual value: {y_test.values[i]}')


# ### 2. SVR
# 

# In[20]:


# do the same with a SVR model

from sklearn.svm import SVR

# create the model
svr_model = SVR()

# train the model
svr_model.fit(X_train, y_train)

# make predictions
y_pred = svr_model.predict(X_test)

# evaluate the model
mse_SVR = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_SVR}')

# Afficher les 10 premières valeurs prédites et les valeurs réelles
for i in range(10):
    print(f'Predicted value: {y_pred[i]:.2f} - Actual value: {y_test.values[i]}')

    


# In[21]:


# hyperparameter tuning with grid search

from sklearn.model_selection import GridSearchCV

# Définir les hyperparamètres à tester

param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}

# Initialiser le modèle SVR
svr_model = SVR()

# Initialiser la recherche sur grille avec les hyperparamètres à tester
grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Entraîner le modèle sur les données d'entraînement
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres
print(grid_search.best_params_)

# Faire des prédictions sur les données de test
y_pred = grid_search.predict(X_test)

# Évaluer les performances du modèle
mse_SVR_tuned = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_SVR_tuned}')

# Afficher les 10 premières valeurs prédites et les valeurs réelles
for i in range(10):
    print(f'Predicted value: {y_pred[i]:.2f} - Actual value: {y_test.values[i]}')

    


# In[22]:


# do the same with a linear regression model

from sklearn.linear_model import LinearRegression

# create the model
linear_model = LinearRegression()

# train the model
linear_model.fit(X_train, y_train)

# make predictions
y_pred = linear_model.predict(X_test)

# evaluate the model
mse_LR = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_LR}')

# Afficher les 10 premières valeurs prédites et les valeurs réelles
for i in range(10):
    print(f'Predicted value: {y_pred[i]:.2f} - Actual value: {y_test.values[i]}')

    


# In[23]:


# do the same with a neural network model

from sklearn.neural_network import MLPRegressor

# create the model
mlp_model = MLPRegressor(random_state=42)

# train the model
mlp_model.fit(X_train, y_train)

# make predictions
y_pred = mlp_model.predict(X_test)

# evaluate the model
mse_NN = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_NN}')

# Afficher les 10 premières valeurs prédites et les valeurs réelles
for i in range(10):
    print(f'Predicted value: {y_pred[i]:.2f} - Actual value: {y_test.values[i]}')


# **Tensorflow Model**
# 

# In[24]:


# create the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])

# change the data type
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)
y_test = y_test.astype(float)

# train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2)

# plot the loss
plt.figure(figsize=(20, 10))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


# make predictions
y_pred = model.predict(X_test)

# evaluate the model
mse_TF = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_TF}')

# Afficher les 10 premières valeurs prédites et les valeurs réelles
for i in range(10):
    print(f'Predicted value: {y_pred[i][0]:.2f} - Actual value: {y_test.values[i]}')
    


# In[26]:


# plot the mse for each model

mse_list = [mse_RF, mse_RF_tuned, mse_SVR, mse_SVR_tuned, mse_LR, mse_NN, mse_TF]

plt.figure(figsize=(20, 10))
ax = sns.barplot(x=['Random Forest', 'Random Forest Tuned', 'SVR', 'SVR Tuned', 'Linear Regression', 'Neural Network', 'TensorFlow'], y=mse_list, palette="rocket") #order=missing_values.sort_values(ascending=False).index
# for i in ax.containers:
#     ax.bar_label(i,)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12)
plt.xticks(rotation=45)
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('MSE for each model')
plt.tight_layout()
plt.show()

