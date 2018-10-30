from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
from Datos import Datos
import numpy as np


dataset=Datos('german.data')


# Encode categorical integer features using a one-hot aka one-of-Kscheme (categorical features)

encAtributos =preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
X = encAtributos.fit_transform(dataset.datos[:,:-1])
Y =dataset.datos[:,-1] 

Nominal = [i for i in range(len(dataset.tipoAtributos[:-1])) if dataset.nominalAtributos[i]]
Continue = [i for i in range(len(dataset.nominalAtributos[:-1])) if not dataset.nominalAtributos[i]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=0)
cols = range(len(dataset.datos[:,0]))
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)

gnb = GaussianNB()
y_predNom = gnb.fit(X_train[:,:], y_train).predict(X_test[:,:])
errorNom = float((y_predNom!=y_test).sum())/len(y_test)
print(errorNom)



mnb = MultinomialNB()
y_predCon = mnb.fit(X_train[:,:], y_train).predict(X_test[:,:	])
errorCon = float((y_predCon!=y_test).sum())/len(y_test)

for rs_train, rs_test in rs.split(cols):
	claseTestG = gnb.fit(dataset.datos[rs_train,:-1], dataset.datos[rs_train,-1]).predict(dataset.datos[rs_test,:-1])
	claseTestM = mnb.fit(dataset.datos[rs_train,:-1], dataset.datos[rs_train,-1]).predict(dataset.datos[rs_test,:-1])


print(errorCon)