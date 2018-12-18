from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from AlgoritmoGenetico import *
from plotModel import plotModel 
import matplotlib.pyplot as plt 
dataset = Datos("example3.data")
alg = ClasificadorAG()
particiones = ValidacionSimple(1,0.9,dataset).creaParticiones(dataset.datos) 
for j in range(1):
	datosTrain = dataset.extraeDatos(particiones[j].indicesTrain)
	datosTest = dataset.extraeDatos(particiones[j].indicesTest)
	alg.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
	#Clasificador.error(datosTest,alg.clasifica(datosTest))
	#"""
	plt.figure()
	plotModel(dataset.datos[particiones[j].indicesTrain,0],dataset.datos[particiones[j].indicesTrain,1],
		  dataset.datos[particiones[j].indicesTrain,-1] !=0, alg,"Frontera",dataset.diccionarios)
	plt.plot(dataset.datos[dataset.datos[:,-1]==0,0],  
	dataset.datos[dataset.datos[:,-1]==0,1],'ro')
	plt.plot(dataset.datos[dataset.datos[:,-1]==1,0],  
	dataset.datos[dataset.datos[:,-1]==1,1],'bo')
	plt.show()
	#"""