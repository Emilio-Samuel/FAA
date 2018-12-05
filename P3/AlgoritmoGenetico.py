import numpy as np
from Clasificador import *
#SOLO 1 PARTICION!
class ClasificadorAG(Clasificador):
	def __init__(self,tamano_poblacion = 50,probabilidad_recombinacion=0.1,probabilidad_mutacion=0.001,proporcion_elitismo=0.05,generaciones=50, max_fitness=0.95):

		self.tamano_poblacion = tamano_poblacion
		self.proporcion_elitismo = proporcion_elitismo
		self.probabilidad_mutacion = probabilidad_mutacion
		self.probabilidad_recombinacion = probabilidad_recombinacion
		self.max_fitness = max_fitness
		self.generaciones = generaciones

	def generar_poblacion(self,nhip):
		poblacion = np.random.randint(0,self.K+1,self.tamano_poblacion*self.natributos).reshape((self.tamano_poblacion,self.natributos))
		poblacion = np.hstack((poblacion,np.random.randint(0,nhip,self.tamano_poblacion).reshape((self.tamano_poblacion,1))))
		return poblacion

	def fitness(self,datosTrain,elem):
		aciertos = 0
		for dato in datosTrain:
			salida = self.discretizar_elemento(dato)
			if(salida[:-1] == elem[:-1] and salida[-1] == elem[-1]) or (salida[:-1] != elem[:-1] and salida[-1] != elem[-1]):
				aciertos +=1
		return aciertos * 1./len(datosTrain)

	def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):

		self.natributos = datosTrain.shape[1] -1
		self.K = np.floor(1+ 3.322*np.log10(len(datosTrain)))
		atributos_continuos = [e for  e,x in enumerate(1- np.array(atributosDiscretos)) if x == 1]
		self.maximos = np.max(datosTrain[:,atributos_continuos],0)
		self.minimos = np.min(datosTrain[:,atributos_continuos],0)
		self.A = (self.maximos - self.minimos)/self.K
		self.poblacion = self.generar_poblacion(l)
		self.fitness_poblacion = np.zeros(len(datosTrain))

		while(self.generaciones > 0 and np.max(self.fitness_poblacion) < self.max_fitness):
			for i in range(len(datosTrain)):
				self.fitness_poblacion[i] = self.fitness(datosTrain,datosTrain[i,:])


	def Cruce(self,elem1, elem2):
		if(np.random.rand(1)[0] > self.probabilidad_recombinacion):
			return [elem1,elem2]
		n = np.random.randint(1,len(elem1)-2,1)[0]

		return [np.hstack((elem1[0:n],elem2[n:])), np.hstack((elem2[0:n],elem1[n:]))]	

	def Mutacion(self,elem):
		mutaciones = np.random.rand(len(elem))
		for i in range(len(elem)-1):
			if mutaciones[i] < self.probabilidad_mutacion:
				elem[i] = np.random.randint(0,self.K+1,1)[0]
		if mutaciones[-1] < self.probabilidad_mutacion:
			elem[-1] = int(not elem[-1])
		return elem

	def discretizar_elemento(self,elem):
		columnas_interes = np.flatnonzero(elem[:-1]) #Todas las columnas menos la clase
		discretizado = np.zeros(len(elem))
		for c in columnas_interes:
			valor = np.ceil((elem[c] - self.minimos[c])/self.A[c])
			discretizado[c] = valor
		return discretizado

	def seleccion_progenitores(self):
		elementos_mantener = -1* self.proporcion_elitismo* len(self.poblacion)
		posiciones = np.argsort(self.fitness_poblacion)
		poblacion = np.copy(self.poblacion[posiciones[elementos_mantener:]])
		

