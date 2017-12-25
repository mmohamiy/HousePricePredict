# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Noviembre/2017
Contenido:
    Uso de XGB, LightGBM y Grid Search en Python para Ciencia de Datos
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
XGBoost y LightGBM:
    https://www.kaggle.com/nschneider/gbm-vs-xgboost-vs-lightgbm
    https://dnc1994.com/2016/03/installing-xgboost-on-windows/
    https://github.com/Microsoft/LightGBM
    https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide

Grid Search:
    http://scikit-learn.org/stable/modules/grid_search.html
    https://www.kaggle.com/tanitter/grid-search-xgboost-with-scikit-learn?scriptVersionId=23363
'''

import pandas as pd
import numpy
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
#from imblearn.metrics import geometric_mean_score
import xgboost as xgb
import lightgbm as lgb

le = preprocessing.LabelEncoder()

#True si cada variable categórica se convierte en varias binarias (tantas como categorías),
#False si solo se convierte la categórica a numérica (ordinal)
binarizar = False

'''
devuelve un DataFrame, los valores perdidos notados como '?' se convierten a NaN,
si no, se consideraría '?' como una categoría más
'''
if not binarizar:
    adult_orig = pd.read_csv('adult.csv')
else:
    adult_orig = pd.read_csv('adult.csv',na_values="?")
    
# devuelve una lista de las características categóricas excluyendo la columna 'class' que contiene la clase
lista_categoricas = [x for x in adult_orig.columns if (adult_orig[x].dtype == object and adult_orig[x].name != 'class')]
if not binarizar:
    adult = adult_orig
else:
    # reemplaza las categóricas por binarias
    adult = pd.get_dummies(adult_orig, columns=lista_categoricas)

# coloco la columna que contiene la clase como última columna por convención
clase = adult['class']
adult.drop(labels=['class'], axis=1,inplace = True)
adult.insert(len(adult.columns), 'class', clase)

# separamos el DataFrame en dos arrays numpy, uno con las características (X) y otro (y) con la clase
# si la última columna contiene la clase, se puede separar así
X = adult.values[:,0:len(adult.columns)-1]
y = adult.values[:,len(adult.columns)-1]
y_bin = le.fit_transform(y) #se convierte a binario para algunas métricas, como f1_score o AUC: '>50K' -> 1 (clase positiva) y '<=50K' -> 0 en adult

'''
Si las variables categóticas tienen muchas categorías, se generarán muchas variables y algunos algoritmos (por ejemplo, SVM) serán
extremadamente lentos. Se puede optar por solo convertirlas a variables numéricas (ordinales) sin binarizar. Esto se haría si no se ha
ejecutado pd.get_dummies() previamente. No funciona si hay valores perdidos notados como NaN
'''
if not binarizar:
    for i in range(0,X.shape[1]):
        if isinstance(X[0,i],str):
            X[:,i] = le.fit_transform(X[:,i])

#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []
    y_prob_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        y_prob = modelo.predict_proba(X[test])[:,1] #la segunda columna es la clase positiva '>50K' en adult
        y_test_bin = le.fit_transform(y[test]) #se convierte a binario para AUC: '>50K' -> 1 (clase positiva) y '<=50K' -> 0 en adult
#        print("Accuracy: {:6.2f}%, F1-score: {:.4f}, G-mean: {:.4f}, AUC: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred)*100 , f1_score(y[test],y_pred,average='macro'), geometric_mean_score(y[test],y_pred,average='macro'), roc_auc_score(y_test_bin,y_prob),tiempo))
        print("Accuracy: {:6.2f}%, F1-score: {:.4f}, AUC: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred)*100 , f1_score(y[test],y_pred,average='macro'), roc_auc_score(y_test_bin,y_prob), tiempo))
        y_test_all = numpy.concatenate([y_test_all,y_test_bin])
        y_prob_all = numpy.concatenate([y_prob_all,y_prob])

    print("")

    return modelo, y_test_all, y_prob_all
#------------------------------------------------------------------------

'''
print("------ XGB...")
clf = xgb.XGBClassifier(n_estimators = 200)
clf, y_test_clf, y_prob_clf = validacion_cruzada(clf,X,y,skf)
#'''

#'''
print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(objective='binary',n_estimators=200,num_threads=2)
lgbm, y_test_lgbm, y_prob_lgbm = validacion_cruzada(lgbm,X,y,skf)
#'''

#'''
print("------ Grid Search...")

#params_xgb = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
#          'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4], 'n_estimators':[50,100,200]}

params_lgbm = {'feature_fraction':[i/10.0 for i in range(3,6)], 'learning_rate':[0.05,0.1], 'num_leaves':[30,50], 'n_estimators':[200]}

grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=1, verbose=1, scoring=make_scorer(f1_score))
grid.fit(X,y_bin)

print("Mejores parámetros:")
print(grid.best_params_)

print("")
gs, y_test_gs, y_prob_gs = validacion_cruzada(grid.best_estimator_,X,y,skf)
#'''

'''
Imputación de valores perdidos:
    https://pypi.python.org/pypi/fancyimpute
    
Desbalanceo de clase:
    https://github.com/scikit-learn-contrib/imbalanced-learn:    

Selección de características con Boruta:    
    https://github.com/scikit-learn-contrib/boruta_py  
    
Instalación:
    pip install <paquete>
    
En Windows, para algunos paquetes (fancyimpute) puede que sea necesario instalar primero esto:
    http://landinghub.visualstudio.com/visual-cpp-build-tools
'''