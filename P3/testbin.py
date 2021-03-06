from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from AlgoritmoGeneticoBin import *

from plotModel import plotModel 
import matplotlib.pyplot as plt 
dataset = Datos("example1.data")
alg = ClasificadorAGB()
particiones = ValidacionBootstrap(1,0.6,dataset).creaParticiones(dataset.datos) 
for j in range(1):
  datosTrain = dataset.extraeDatos(particiones[j].indicesTrain)
  datosTest = dataset.extraeDatos(particiones[j].indicesTest)
  alg.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
  print(alg.fitness(datosTest,alg.regla))
  print(Clasificador.error(datosTest,alg.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios)))
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