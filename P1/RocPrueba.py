from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
TPR = [0]
FPR = [0]
dataset=Datos("tic-tac-toe.data")
#Calculamos la matriz roc:
particiones = ValidacionSimple(1,0.3,dataset).creaParticiones(dataset.datos)
for x in particiones:
  cnv = ClasificadorNaiveBayes()
  datosTrain = dataset.extraeDatos(x.indicesTrain)
  datosTest = dataset.extraeDatos(x.indicesTest)

  cnv.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
  NL = cnv.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios)
  matriz = Clasificador.matrizConfusion(datosTest,NL,0)
  print(matriz)
  z,y = cnv.Curva_roc(datosTest,dataset.nominalAtributos,dataset.diccionarios)

  plt.plot(y,z,'o')
  ax = np.linspace(0,1,num=100)
  plt.plot(ax,ax)
  plt.show()