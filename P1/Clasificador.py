from abc import ABCMeta,abstractmethod
import numpy as np

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
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error    
  	return sum(clasificacion != datostest[:,c-1])/len(pred)
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
    pass
       
  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

 
  def __init__(self):
    self.tablas = []

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario,laplace = False):
	
    [f,c] = datostrain.shape
    hipotesis = diccionario[-1]
    l = len(hipotesis)
    continuos = [[] for _ in range(l)]
    for i in range(c):
      if(atributosDiscretos[i]):
        self.tablas.append(np.zeros((len(diccionario[i]),l)))
      else:
        self.tablas.append(np.zeros((2,l)))
    #tablas = [np.zeros((len(hipotesis),len(x))) for x in diccionario if len(x) != 0]
    for i in range(f):
      for j in range(c):
        if atributosDiscretos[j]:
          self.tablas[j][int(datostrain[i,j]),int(datostrain[i,c-1])] +=1
        else:
          continuos[int(datostrain[i,c-1])].append(datostrain[i,j])

    if laplace :
      for i in range(c-1):
        if 0 in self.tablas[i] :
          #print( self.tablas[i])
          self.tablas[i] =  self.tablas[i]+1
    i=0
    for h in range(l):
      for j in range(c-1):
        if not atributosDiscretos[j]:
          self.tablas[j][0,h] = np.mean(continuos[i])
          self.tablas[j][1,h] = np.std(continuos[i])
          i += 1

    print(self.tablas)
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):

    [f,c] = datostest.shape
    hipotesis = diccionario[-1]
    l = len(hipotesis)
    clasificacion = []
    for i in range(f):
      p = np.zeros((1,l))
      for h in range(l):
        for j in range(c-2):
          if(atributosDiscretos[j]):
            p[0,h] +=  self.tablas[j][int(datostest[i,j]),h]/sum(self.tablas[j][:,h])
        #print(self.tablas[-1][h,h]/sum(sum(self.tablas[-1])))
        p[0,h] += self.tablas[-1][h,h]/sum(sum(self.tablas[-1]))
      #print(p,np.max(p),np.where(p == np.max(p)))
      clasificacion.append(np.where(p == np.max(p))[1][0])
    print(sum(clasificacion != datostest[:,c-1])/f)
    return clasificacion



    

  