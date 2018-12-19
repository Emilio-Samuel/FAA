from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from AlgoritmoGeneticoBin import *
dataset = Datos("example4.data")
alg = ClasificadorAGB()
particiones = ValidacionBootstrap(1,0.1,dataset).creaParticiones(dataset.datos) 
for j in range(1):
  datosTrain = dataset.extraeDatos(particiones[j].indicesTrain)
  datosTest = dataset.extraeDatos(particiones[j].indicesTest)
  alg.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
  print(alg.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios))