# -*- coding: utf8 -*-
from abc import ABCMeta,abstractmethod
import numpy as np
from  scipy.stats import norm 
import matplotlib.pyplot as plt
class Clasificador(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  @staticmethod
  def error(datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error    
  	return sum(pred != datos[:,-1])/len(pred)

  # Obtiene el numero de falsos positivos yls falsos negativos para calcular la matriz de confusion
  # TODO: implementar
  @staticmethod
  def matrizConfusion(datos,pred,clasePositiva):
    #print(clasePositiva)
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula la matriz de confusion

    tp = sum([(x[0] == clasePositiva and x[1] == clasePositiva) for x in  zip(datos[:,-1],pred)])
    fp = sum([(x[1] == clasePositiva and x[0] != clasePositiva) for x in  zip(datos[:,-1],pred)])
    tn = sum([(x != clasePositiva and y != clasePositiva) for x,y in  zip(datos[:,-1],pred)])
    fn = sum([(x == clasePositiva and y != clasePositiva) for x,y in  zip(datos[:,-1],pred)])
    #print("Valores",tp,fp,tn,fn)
    TPR = 0 if tp == 0 else tp/(tp+fn)
    FPR = 0 if fp == 0 else fp/(tn+fp)
    return [[TPR, FPR],[fn/(tp+fn), tn/(tn+fp)]]

  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
    pass
  def Curva_roc(self, datostest,atributosDiscretos,diccionario):
    probs = self.clasifica(datostest,atributosDiscretos,diccionario, True)
    valores = np.copy(probs)
    valores.sort()
    TPR = [0]
    FPR = [0]
    for i in valores:
      clasificacion = probs < i
      #print("\n",clasificacion)

      matriz = Clasificador.matrizConfusion(datostest,clasificacion,True)
      #print(matriz)
      TPR.append(matriz[0][0])
      FPR.append(matriz[0][1])
    #print(TPR,FPR)
    TPR.append(1)
    FPR.append(1)
    return TPR,FPR


##############################################################################

class ClasificadorNaiveBayes(Clasificador):

 
  def __init__(self):
    self.tablas = []

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario,laplace = False):
	
    [f,c] = datostrain.shape
    hipotesis = diccionario[-1]
    l = len(hipotesis)
    continuos = [[] for _ in range(c-1)]
    for i in range(c):
      if(atributosDiscretos[i]):
        self.tablas.append(np.zeros((len(diccionario[i]),l)))
      else:
        self.tablas.append(np.zeros((2,l)))
    #tablas = [np.zeros((len(hipotesis),len(x))) for x in diccionario if len(x) != 0]
    #Contamos las apariencias de cada dato
    #En las columnas las hipotesis
    #Filas los datos discretos o media y varianza si son continuos
    for i in range(f):
      for j in range(c):
        if atributosDiscretos[j]:
          self.tablas[j][int(datostrain[i,j]),int(datostrain[i,c-1])] +=1
        else:
          continuos[j].append(datostrain[i,j])
    #Miramos si hay algun 0 en alguna tabla, si lo hay añadimos 1 a todo

    if laplace :
      for i in range(c):
        if 0 in self.tablas[i] and atributosDiscretos[i]:
          #print( self.tablas[i])
          self.tablas[i] =  self.tablas[i]+1
    i=0
    #Ponemos las medias y varianzas.
    for h in range(l):
      for j in range(c-1):
        if not atributosDiscretos[j]:
          self.tablas[j][0,h] = np.mean(continuos[j])
          self.tablas[j][1,h] = np.std(continuos[j])

  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario,prob = False):

    [f,c] = datostest.shape
    hipotesis = diccionario[-1]
    l = len(hipotesis)
    clasificacion = []
    probs = np.zeros(f)
    for i in range(f):
      p = np.ones(l)#probabilidades de cada hipotesis
      for h in range(l):
        if(sum(self.tablas[0][:,h]) == 0):
          p[h] = 0
          continue;

        for j in range(c-1):
          if(atributosDiscretos[j]):
            p[h] = p[h] * self.tablas[j][int(datostest[i,j]),h]/sum(self.tablas[j][:,h])#Hacemos los a posteriori
          else:
            p[h] = p[h] * norm.pdf(datostest[i,j], loc = self.tablas[j][0,h],scale = self.tablas[j][1,h])

        p[h] =  p[h] * self.tablas[-1][h,h]/sum(sum(self.tablas[-1]))
    
      clasificacion.append(np.where(p == np.max(p))[0][0])
      probs[i] = p[0]/np.sum(p)
    if prob:
      return probs
    return clasificacion


  def validacion(self,particionado,dataset,clasificador,seed=None):
    particiones = particionado.creaParticiones(dataset.datos)  
    for x in particiones:
      datosTrain = dataset.extraeDatos(x.indicesTrain)
      datosTest = dataset.extraeDatos(x.indicesTest)
      self.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)  
      self.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios)