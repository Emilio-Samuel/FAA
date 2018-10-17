from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
dataset=Datos('german.data')
estrategia = []
estrategia.append(ValidacionSimple(20,0.05,dataset))
estrategia.append(ValidacionCruzada(20,dataset))
estrategia.append(ValidacionBootstrap(20,dataset))
erroresNV = [[],[],[]]
erroresL = [[],[],[]]
estrategias = ["ValidacionSimple","ValidacionCruzada","ValidacionBootstrap"]
print("Iteracion \tError Laplace \t Error sin Laplace\n")
for i in range(3):
	print(estrategias[i])
	particiones = estrategia[i].creaParticiones(dataset.datos)	
	j = 0
	for x in particiones:
		j +=1
		cnv = ClasificadorNaiveBayes()
		datosTrain = dataset.extraeDatos(x.indicesTrain)
		datosTest = dataset.extraeDatos(x.indicesTest)
		#datosNominales = [x for x in range(len(dataset.nominalAtributos)) if dataset.nominalAtributos[x]]
		cnv.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
		NL = cnv.clasifica(datosTest, 
			dataset.nominalAtributos,dataset.diccionarios)
		cnvL = ClasificadorNaiveBayes()
		cnvL.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios, True)

		L = cnvL.clasifica(datosTest, 
			dataset.nominalAtributos,dataset.diccionarios)
		errorNV = Clasificador.error(datosTest,L)
		errorL = Clasificador.error(datosTest,NL)
		erroresNV[i].append(errorNV)
		erroresL[i].append(errorL)
		print("{j}\t{l}\t{nl}".format(j = j,l = errorL,nl = errorNV),end="\r")

	print("                                                                 ",end="\r")
	print("\tError medio:\n")
	#print(erroresL[i])
	mediaL = np.mean(erroresL[i])
	mediaNv = np.mean(erroresNV[i])
	print("\t{l}\t{nl}".format(l = mediaL, nl = mediaNv))
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
