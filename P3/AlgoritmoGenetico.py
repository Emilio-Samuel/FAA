import numpy as np
from Clasificador import *
import time
from scipy.stats import mode
#SOLO 1 PARTICION!
class ClasificadorAG(Clasificador):
	def __init__(self,tamano_poblacion = 100,probabilidad_recombinacion=0.75,probabilidad_mutacion=0.4,proporcion_elitismo=0.05,generaciones=100, max_fitness=0.99):

		self.tamano_poblacion = tamano_poblacion
		self.proporcion_elitismo = proporcion_elitismo
		self.probabilidad_mutacion = probabilidad_mutacion
		self.probabilidad_recombinacion = probabilidad_recombinacion
		self.max_fitness = max_fitness
		self.generaciones = generaciones
		self.mejora = 0
		self.tiempo_sin_mejora = 0
		self.Generacion = None

	def generar_poblacion(self,nhip,cota_reglas = 50):
		poblacion = []
		for i in range(self.tamano_poblacion):
			n_reglas = np.random.randint(1,cota_reglas,1)[0]
			individuo = np.random.randint(0,self.K+1,n_reglas*self.natributos).reshape((n_reglas,self.natributos))
			individuo = np.hstack((individuo,np.random.randint(0,nhip,n_reglas).reshape((n_reglas,1))))
			poblacion.append(individuo)
		return poblacion

	def fitness(self,datosTrain,elem):
		aciertos = 0
		for dato in datosTrain:
			salida = self.discretizar_elemento(dato)
			clases = []
			for regla in elem:
				flag = True
				for i in range(len(regla)-1):
					if regla[i]!=0 and salida[i] != regla[i]:
						flag = False
						break
				if flag:
					clases.append(regla[-1])
				else:
					clases.append(int(not regla[-1]))
			#print(clases)
			
			if mode(clases)[0][0] == dato[-1]:
				aciertos += 1
				
		return aciertos * 1./len(datosTrain)

	def entrenamiento(self,datosTrain,atributosDiscretos,diccionario,verbose=False):

		self.natributos = datosTrain.shape[1] -1
		self.K = np.floor(1+ 3.322*np.log10(len(datosTrain)))
		atributos_continuos = [e for  e,x in enumerate(1- np.array(atributosDiscretos)) if x == 1]
		self.maximos = np.max(datosTrain[:,atributos_continuos],0)
		self.minimos = np.min(datosTrain[:,atributos_continuos],0)
		self.A = (self.maximos - self.minimos)/self.K
		self.poblacion = self.generar_poblacion(len(diccionario[-1]))
		self.fitness_poblacion = np.zeros(self.tamano_poblacion)
		#print(self.poblacion)
		for i in range(self.tamano_poblacion):
			self.fitness_poblacion[i] = self.fitness(datosTrain,self.poblacion[i])

		while(self.generaciones > 0 and np.max(self.fitness_poblacion) < self.max_fitness and self.tiempo_sin_mejora < 15):
			self.poblacion = self.seleccion_progenitores()
			self.tamano_poblacion = len(self.poblacion)
			for i in range(self.tamano_poblacion - 1):
				individuo1 = self.poblacion.pop(i)
				individuo2 = self.poblacion.pop(i)
				individuos = self.Cruce(individuo1,individuo2)
				self.poblacion.append(individuos[0])
				self.poblacion.append(individuos[1])
				i += 1

			for i in range(self.tamano_poblacion):
				self.poblacion[i] = self.Mutacion(self.poblacion[i])

			self.poblacion.extend(self.elitismo_a_mantener)
			self.tamano_poblacion = len(self.poblacion)
			for i in range(self.tamano_poblacion):
				self.fitness_poblacion[i] = self.fitness(datosTrain,self.poblacion[i])

			self.generaciones -= 1
			mejora = np.max(self.fitness_poblacion)
			if mejora > self.mejora:
				self.tiempo_sin_mejora = 0
				self.mejora = mejora
			else:
				self.tiempo_sin_mejora += 1
			print(self.generaciones,mejora,end="\r")
			if verbose:
				if self.Generacion is None:
					self.Generacion = self.generaciones
					self.Mejores = self.mejora
					self.fitness_medio = np.sum(self.fitness_poblacion)*1./self.tamano_poblacion
				else:
					self.Generacion = np.hstack((self.Generacion,self.generaciones))
					self.Mejores = np.hstack((self.Mejores,self.mejora))
					self.fitness_medio = np.hstack((self.fitness_medio,np.sum(self.fitness_poblacion)*1./self.tamano_poblacion))

		n_ganador = np.argsort(self.fitness_poblacion)[-1]
		self.regla = self.poblacion[n_ganador]
		print(self.generaciones)
		print(self.regla)
		print(self.fitness(datosTrain,self.regla))
		return
	def clasifica(self,datostest,atributosDiscretos,diccionario):
		clasificaciones =[]
		for dato in datostest:
			salida = self.discretizar_elemento(dato)
			clases = []
			for regla in self.regla:
				flag = True
				for i in range(len(regla)-1):
					if regla[i]!=0 and salida[i] != regla[i]:
						flag = False
						break
				if flag:
					clases.append(regla[-1])
				else:
					clases.append(int(not regla[-1]))
			
			clasificaciones.append(mode(np.array(clases))[0][0]) 
		return np.array(clasificaciones)
	def Cruce(self,elem1, elem2):

		if len(elem1.shape) == 1:
			elem1 = elem1.reshape((1,elem1.shape[0]))
		if len(elem2.shape) == 1:
			elem2 = elem2.reshape((1,elem2.shape[0]))

		if(np.random.rand(1)[0] > self.probabilidad_recombinacion):
			return [elem1,elem2]

		if elem2.shape[0] == 1 or elem1.shape[0] == 1:

			n = np.random.randint(1,len(elem1[0,:]),1)[0]

			return [np.hstack((elem1[0,0:n],elem2[0,n:])), np.hstack((elem2[0,0:n],elem1[0,n:]))]	
		
		else:

			
			n = np.random.randint(1,np.min((elem1.shape[0],elem2.shape[0])),1)[0]
			
			return [np.vstack((elem1[0:n,:],elem2[n:,:])), np.vstack((elem2[0:n,:],elem1[n:,:]))]
		

	def Mutacion(self,elem):
		if len(elem.shape) == 1:
			elem = elem.reshape((1,elem.shape[0]))
		for fila in elem:
			mutaciones = np.random.rand(len(fila))
			for i in range(len(fila)-1):
				if mutaciones[i] < self.probabilidad_mutacion:
					fila[i] = np.random.randint(0,self.K+1,1)[0]
			if mutaciones[-1] < self.probabilidad_mutacion:
				fila[-1] = int(not fila[-1])
		if np.random.rand(1)[0] < self.probabilidad_mutacion:
			nueva_regla = np.random.randint(0,self.K+1,self.natributos)
			nueva_regla = np.hstack((nueva_regla,np.random.randint(0,2,1)))
			elem = np.vstack((elem,nueva_regla)) 
		if np.random.rand(1)[0] < self.probabilidad_mutacion:
			n = np.random.randint(1,elem.shape[0]-1,1)[0]
			elem = np.vstack((elem[0:n],elem[n+1:]))

		return elem

	def discretizar_elemento(self,elem):
		columnas_interes = np.flatnonzero(elem[:-1]) #Todas las columnas menos la clase
		discretizado = np.zeros(len(elem))
		for c in columnas_interes:
			valor = np.ceil((elem[c] - self.minimos[c])/self.A[c])
			discretizado[c] = valor
		discretizado[-1] = elem[-1]
		return discretizado.astype(int)
	def seleccion_progenitores(self):
		elementos_mantener = int(np.floor(-1* self.proporcion_elitismo* len(self.poblacion)))
		
		posiciones = np.argsort(self.fitness_poblacion)
		poblacion = []
		self.elitismo_a_mantener = []
		for e in posiciones[elementos_mantener:]:
			self.elitismo_a_mantener.append(self.poblacion[e])
		probabilidades = np.cumsum(self.fitness_poblacion)/np.sum(self.fitness_poblacion)
		probabilidades = np.hstack((0,probabilidades))
		decision = np.random.rand(len(self.poblacion) + elementos_mantener)

		for p in decision:
			#print(np.where(probabilidades < p)[0], p)
			elegido = np.where(probabilidades < p)[0][0]
			poblacion.append(self.poblacion[elegido])
		return poblacion