from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
from sklearn import preprocessing 
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import train_test_split
import warnings

import warnings
warnings.filterwarnings("ignore")
datasets = ["balloons.data","tic-tac-toe.data","german.data"]
particiones = [20,10,5]
estrategia = []
for z in range(3):
	print(datasets[z])
	dataset=Datos(datasets[z])
	#Anadimos las estrategias para comparar
	
	estrategia.append(ValidacionSimple(particiones[z],0.1,dataset))
	estrategia.append(ValidacionCruzada(particiones[z],dataset))
	estrategia.append(ValidacionBootstrap(particiones[z],dataset))
	#Listas de errores para poder calcular estadisticos
	erroresNV = [[],[],[]]
	erroresL = [[],[],[]]
	erroresSkG = [[],[],[]]
	erroresSkN = [[],[],[]]
	estrategias = ["ValidacionSimple","ValidacionCruzada","ValidacionBootstrap"]
	print("Iteracion \tError Laplace \t Error sin Laplace \t ErrorSklearn\n")
	#Procesamiento de los datos para comparar con paquete sklearn
	encAtributos = preprocessing.OneHotEncoder(
		categorical_features = dataset.nominalAtributos[:-1],
		sparse = False)
	X = encAtributos.fit_transform(dataset.datos[:,:-1])
	Y = dataset.datos[:,-1]

	for i in range(3):
		print(estrategias[i])

		particion = estrategia[int(i+z*3)].creaParticiones(dataset.datos)	
		j = 0
		for x in particion:
			j +=1
			cnv = ClasificadorNaiveBayes()
			datosTrain = dataset.extraeDatos(x.indicesTrain)
			#print(x.indicesTrain)
			#print(x.indicesTest)
			datosTest = dataset.extraeDatos(x.indicesTest)
			#datosNominales = [x for x in range(len(dataset.nominalAtributos)) if dataset.nominalAtributos[x]]
			cnv.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
			NL = cnv.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios)
			cnvL = ClasificadorNaiveBayes()
			cnvL.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios, True)

			L = cnvL.clasifica(datosTest, dataset.nominalAtributos,dataset.diccionarios)
			errorNV = Clasificador.error(datosTest,L)
			errorL = Clasificador.error(datosTest,NL)

			erroresNV[i].append(errorNV)
			erroresL[i].append(errorL)
			#Comprobamos con sklearn
			X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=None)
			#print(X_train)
			gnb = GaussianNB()
			y_pred = gnb.fit(X_train,y_train).predict(X_test)
			errorSkG = (y_test != y_pred).sum()/len(y_pred)
			erroresSkG[i].append(errorSkG)
			gnb = MultinomialNB()
			y_pred = gnb.fit(X_train,y_train).predict(X_test)
			errorSkN = (y_test != y_pred).sum()/len(y_pred)
			erroresSkN[i].append(errorSkN)
			print("{j}\t{l}\t{nl}\t{e}\t{e1}".format(j = j,l = errorL,nl = errorNV,e = errorSkG,e1 = errorSkN),end="\r")

		print("                                                                                                                  ",end="\n")
		print("\tSegundo cuantil del error:\n")
		#print(erroresL[i])
		mediaL = np.median(erroresL[i])
		mediaNv = np.median(erroresNV[i])
		mediaSk = np.median(erroresSkG[i])
		mediaSkN = np.median(erroresSkN[i])
		print("\t{l}\t{nl}\t{e}\t{e1}\n".format(l = mediaL, nl = mediaNv,e = mediaSk,e1=mediaSkN))
#estrategia1 = ValidacionBootstrap(1,20,dataset)
#particiones = estrategia1.creaParticiones(dataset.datos)
#for x in particiones:
#	print(dataset.extraeDatos(x.indicesTest))
#	print(dataset.extraeDatos(x.indicesTrain))

"""
estrategia1 = ValidacionCruzada(3,dataset)
particiones1 = estrategia1.creaParticiones(dataset.datos)
for x in particiones1:
	print ("\n")
	print(dataset.extraeDatos(x.indicesTest))
	print(dataset.extraeDatos(x.indicesTrain))
"""
