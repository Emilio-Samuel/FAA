from Datos import Datos
from EstrategiaParticionado import *

dataset=Datos('balloons.data')
#estrategia = ValidacionSimple(1,0.4,dataset)
#particiones = estrategia.creaParticiones(dataset.datos)
#for x in particiones:
#	print(dataset.extraeDatos(x.indicesTest))
#	print(dataset.extraeDatos(x.indicesTrain))

estrategia1 = ValidacionCruzada(3,dataset)
particiones1 = estrategia1.creaParticiones(dataset.datos)
for x in particiones1:
	print "\n"
	print(dataset.extraeDatos(x.indicesTest))
	print(dataset.extraeDatos(x.indicesTrain))