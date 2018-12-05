from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from AlgoritmoGenetico import *
dataset = Datos("wdbc.data")
alg = AlgoritmoGenetico(tamano_poblacion = 50,probabilidad_recombinacion=0.1,probabilidad_mutacion=0.001,
    proporcion_elitismo=0.05,generaciones=50, max_fitness=0.95)
particiones = ValidacionSimple(1,0.3,dataset).creaParticiones(dataset.datos) 
for j in range(1):
  datosTrain = dataset.extraeDatos(particiones[j].indicesTrain)
  datosTest = dataset.extraeDatos(particiones[j].indicesTest)
  alg.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)