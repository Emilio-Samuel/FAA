from Datos import Datos
from EstrategiaParticionado import *

dataset=Datos('balloons.data')
estrategia = ValidacionSimple(1,0.4,dataset)
particiones = estrategia.creaParticiones(dataset.datos)
for x in particiones:
	print(dataset.extraeDatos(x.indicesTest))
	print(dataset.extraeDatos(x.indicesTrain))

estrategia1 = ValidacionCruzada(1,0.4,dataset)