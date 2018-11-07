from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
import matplotlib.pyplot as plt

datasets = ["balloons.data","tic-tac-toe.data","german.data"]
ax = np.linspace(0,1,num=100)
for z in range(3):
    TPR = []
    FPR = [] 
    dataset=Datos(datasets[z])
    plt.subplot(1,3,z+1)
    plt.title(datasets[z])
    plt.plot(ax,ax)
    #Calculamos la matriz roc:
    particiones = ValidacionBootstrap(20,dataset).creaParticiones(dataset.datos)
    for x in particiones:
      cnv = ClasificadorNaiveBayes()
      datosTrain = dataset.extraeDatos(x.indicesTrain)
      datosTest = dataset.extraeDatos(x.indicesTest)

      cnv.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
      NL = cnv.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios)
      matriz = Clasificador.matrizConfusion(datosTest,NL,0)
      #print(matriz)
      X,y = cnv.Curva_roc(datosTest,dataset.nominalAtributos,dataset.diccionarios)
      plt.plot(y,X)
      
    
plt.show()