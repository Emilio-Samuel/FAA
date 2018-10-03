from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado
from ValidacionSimple import ValidacionSimple
from Particion import Particion
dataset=Datos('../balloons.data')
estrategia = ValidacionSimple(1,0.4,dataset)
estrategia.creaParticiones()
for x in estrategia.particiones:
	print(dataset.extraeDatos(x.indicesTest))
	print(dataset.extraeDatos(x.indicesTrain))