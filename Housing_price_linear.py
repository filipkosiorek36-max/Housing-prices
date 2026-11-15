import kagglehub
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

sklearn.set_config(transform_output='pandas')
#Pobranie danych z Kaggle
path = kagglehub.competition_download('home-data-for-ml-course')
print("Path to competition files:", path)

path_to_train = os.path.join(path, 'train.csv')
path_to_test = os.path.join(path, 'test.csv')

train_data = pd.read_csv(path_to_train)
test_data = pd.read_csv(path_to_test)

#Musze oddzielic juz dane docelowe oraz Id zeby nie zosaly przeskalowane przez co dane ktore przygotowujemy to tylko X
Y_train = np.log1p(train_data['SalePrice'])
Id_train = train_data['Id']
Id_test = test_data['Id']
train_data = train_data.drop(columns= ['SalePrice', 'Id'])
test_data = test_data.drop(columns= ['Id'])

#Przygotowaanie danych do modelu
train_data_numb = train_data.select_dtypes(include= ['number'])
test_data_numb = test_data.select_dtypes(include = ['number'])

train_data_word = train_data.select_dtypes(include = ['object', 'category', 'str'])
test_data_word = test_data.select_dtypes(include = ['object', 'category', 'str'])

#usuwanie brakow slow i zmiana na liczby
train_data_word = train_data_word.fillna('None')
test_data_word = test_data_word.fillna('None')

encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
train_data_word = encoder.fit_transform(train_data_word)
test_data_word = encoder.transform(test_data_word)


#Zmiana NaN na mediane do regresji liniowej
imputer_numb = SimpleImputer(strategy = 'median')

train_data_numb = imputer_numb.fit_transform(train_data_numb)
test_data_numb = imputer_numb.transform(test_data_numb)

# Skalowanie danych 
scaler = StandardScaler()
train_data_numb = scaler.fit_transform(train_data_numb)
test_data_numb = scaler.transform(test_data_numb)

#spajanie danych do regresji (juz nasze dane sa podzielone na X i Y)
X_train = pd.concat([train_data_numb, train_data_word], axis = 1)
X_test = pd.concat([test_data_numb, test_data_word], axis = 1)

#Usuwanie nie potrzebnych kolumn oraz wybranie modelu do pipeline any uniknac leaku danynch
helper = LassoCV(cv = 5, random_state = 42)
alpha = np.logspace(-3, 3, 100)

pipeline_lin = Pipeline([
    ('selector', SelectFromModel(helper)),
    ('model', RidgeCV(alphas = alpha, cv = 5)),
])
#Wyliczenie bledu RMSE
score_lin = cross_val_score(
    pipeline_lin, X_train, Y_train,
    scoring = 'neg_root_mean_squared_error',
    cv = 5
)

rmse_lin = -score_lin.mean()
std_lin = score_lin.std()
print(f'RMSE ridge regression wynosi: {rmse_lin:.4f} , a odchylenie standardowe ( +/- {score_lin.std():.4f})')

