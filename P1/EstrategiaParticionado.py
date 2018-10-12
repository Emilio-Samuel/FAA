# -*- coding: utf-8 -*-
from abc import ABCMeta,abstractmethod
import random
import numpy as np
class Particion():

	# Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones  
	def __init__(self):
		self.indicesTrain=[]
		self.indicesTest=[]

	def __init__(self,indicesTrain,indicesTest):
		self.indicesTest = indicesTest
		self.indicesTrain = indicesTrain

#####################################################################################################

class EstrategiaParticionado(object):
	
	# Clase abstracta
	__metaclass__ = ABCMeta
	
	# Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones, listaParticiones. Se pasan en el constructor 
	def __init__(self,nombreEstrategia, numeroParticiones,seed=None):
			self.nombreEstrategia = nombreEstrategia
			self.numeroParticiones = numeroParticiones
			random.seed(seed)
	@abstractmethod
	# TODO: esta funcion deben ser implementadas en cada estrategia concreta  
	def creaParticiones(self,datos,seed=None):
		pass
	

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

	# Constructor que llama al de la superclase.
	def __init__(self,numeroParticiones, porcentaje,seed=None):
		super(ValidacionSimple,self).__init__("Validacion simple",numeroParticiones)
		self.porcentaje = porcentaje
	
	# Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
	# Devuelve una lista de particiones (clase Particion)
	# TODO: implementar
	def creaParticiones(self,datos,seed=None):
		random.seed(seed)
		particiones = []
		ntot = len(datos)
		n = int(ntot*self.porcentaje)
		for i in range(self.numeroParticiones):
			x = list(range(ntot))
			random.shuffle(x)
			p = Particion(x[:n],x[(n+1):])
			particiones.append(p)
		return particiones

			
			
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
	

	def __init__(self,numFolds,datos,seed=None):
		super().__init__("Validacion simple",numFolds)
		self.numFolds = numFolds

	# Crea particiones segun el metodo de validacion cruzada.
	# El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
	# Esta funcion devuelve una lista de particiones (clase Particion)
	# TODO: implementar
	def creaParticiones(self,datos,seed=None):   
		random.seed(seed)
		particiones = []
		ntot = len(datos)
		x = list(range(ntot))
		n = floor(ntot/self.numFolds)
		for i in range((self.numFolds)):
			p = Particion(x[n*i:n*(i+1)],x[:n*i]+x[(n*i)+1:])
			particiones.append(p)
		return particiones
		pass
		

#####################################################################################################      
class ValidacionBootstrap(EstrategiaParticionado):
	

	def __init__(self,numFolds,datos,seed=None):
		super().__init__("Validacion simple",numFolds)
		self.tamano = datos
	# Crea particiones segun el metodo de validacion por bootstrap.
	# Esta funcion devuelve una lista de particiones (clase Particion)
	# TODO: implementar
	def creaParticiones(self,datos,seed=None):   
		random.seed(seed)
		l = len(datos)
		particiones = []
		for i in range(self.numeroParticiones):
			p1 = list(set(np.random.randint(l,size=self.tamano)))
			ints = [i for i in range(l) if i not in p1]
			particiones.append(Particion(p1,ints))
		return particiones
		
