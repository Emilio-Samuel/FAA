from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np

dataset=Datos("balloons.data")
cnv = ClasificadorNaiveBayes()
cnv.entrenamiento(dataset.extraeDatos(range(20)),dataset.nominalAtributos,dataset.diccionarios, True)
L = cnv.clasifica(dataset.extraeDatos(range(20)), dataset.nominalAtributos,dataset.diccionarios)
print(L)