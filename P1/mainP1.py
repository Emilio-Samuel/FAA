from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
dataset=Datos('balloons.data')
estrategia = ValidacionSimple(1,0.4,dataset)
particiones = estrategia.creaParticiones(dataset.datos)
for x in particiones:
	print(dataset.extraeDatos(x.indicesTest))
	print(dataset.extraeDatos(x.indicesTrain))
cnv = ClasificadorNaiveBayes()
#datosNominales = [x for x in range(len(dataset.nominalAtributos)) if dataset.nominalAtributos[x]]
cnv.entrenamiento(dataset.extraeDatos(x.indicesTrain),dataset.nominalAtributos,dataset.diccionarios)
print(cnv.clasifica(dataset.extraeDatos(x.indicesTest), 
	dataset.nominalAtributos,dataset.diccionarios))
cnvL = ClasificadorNaiveBayes()
cnvL.entrenamiento(dataset.extraeDatos(x.indicesTrain),dataset.nominalAtributos,dataset.diccionarios, True)

print(cnvL.clasifica(dataset.extraeDatos(x.indicesTest), 
	dataset.nominalAtributos,dataset.diccionarios))
#estrategia1 = ValidacionBootstrap(1,20,dataset)
#particiones = estrategia1.creaParticiones(dataset.datos)
#for x in particiones:
#	print(dataset.extraeDatos(x.indicesTest))
#	print(dataset.extraeDatos(x.indicesTrain))