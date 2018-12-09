from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from AlgoritmoGenetico import *
dataset = Datos("wdbc.data")
alg = ClasificadorAG()
particiones = ValidacionBootstrap(1,0.1,dataset).creaParticiones(dataset.datos) 
for j in range(1):
  datosTrain = dataset.extraeDatos(particiones[j].indicesTrain)
  datosTest = dataset.extraeDatos(particiones[j].indicesTest)
  alg.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
  print(alg.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios))