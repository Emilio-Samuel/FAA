import numpy as np
from Clasificador import *
from scipy.stats import mode
import copy
#SOLO 1 PARTICION!
#100 individuos-100 generaciones
#100 individuos-200 generaciones
#200 individuos-100 generaciones
#200 individuos-200 generaciones
class ClasificadorAGB(Clasificador):
	def __init__(self,tamano_poblacion = 50,probabilidad_recombinacion=0.7,probabilidad_mutacion=0.3,proporcion_elitismo=0.05,generaciones=100, max_fitness = 0.99):

		self.tamano_poblacion = tamano_poblacion
		self.proporcion_elitismo = proporcion_elitismo
		self.probabilidad_mutacion = probabilidad_mutacion
		self.probabilidad_recombinacion = probabilidad_recombinacion
		self.max_fitness = max_fitness
		self.generaciones = generaciones
		self.mejora = 0
		self.tiempo_sin_mejora = 0
		self.Generacion = None

	def generar_poblacion(self,nhip):
		#Inicializamos una poblacion vacia
		poblacion=[]
		aux = []		#Creamos un individuo
		for cromosoma in range(self.tamano_poblacion):
			individuo = []
			#Inicialmente un individuo tiene entre 1 y 10 reglas
			num_reglas = np.random.randint(1,20)
			for i in range(num_reglas):
				regla = []
				for j in range(self.natributos):
					#Para cada atributo de la regla creamos un array binario
					n= np.random.randint(0,2)
					rangos = 2**(self.K-1)-1
					valoresAtrib = np.random.permutation(np.asarray([int(d) for d in np.binary_repr(rangos, width=self.K)]))
					regla.append(valoresAtrib)
				#Le annadimos una clase aleatoria y metemos la regla en nuestro individuo
				regla.append(np.random.randint(0,2))
				individuo.append(regla)
			#Lo annadimos a la poblacion
			poblacion.append(individuo)
		return poblacion


	def fitness(self,datosTrain,elem):
		aciertos = 0
		num_reglas = len(elem)
		
		for dato in datosTrain:
			#Hallo el intevalo al que pertenece cada atributo y annado la clase al final
			salida = self.discretizar_elemento(dato)
			flag = False
			clase = []
			for regla in elem:
				flag_atributos = True
				for atributo in range(self.natributos):

					opciones = np.array(np.where(np.asarray(regla[atributo]) == 1)).ravel()
					if list(opciones)==[]:
						continue
					if not np.any(opciones == salida[atributo]):
						flag_atributos = False
						break
				if flag_atributos == True:
					clase.append(regla[-1])
			if len(clase) == 0:
				clase.append(self.priori)
			if mode(clase)[0][0] == salida[-1]:
				aciertos += 1

		return aciertos*1./len(datosTrain)

	def entrenamiento(self,datosTrain,atributosDiscretos,diccionario,verbose = False):
		priori = np.bincount(datosTrain[:,-1].astype(np.int64))
		#priori = np.mean(datosTrain)
		self.priori = np.argmax(priori)

		self.natributos = max(datosTrain.shape[1] -12,2)
		self.K = np.floor(1+ 3.322*np.log10(len(datosTrain)))
		self.K = self.K.astype(np.int64)
		atributos_continuos = [e for  e,x in enumerate(1- np.array(atributosDiscretos)) if x == 1]
		self.maximos = np.max(datosTrain[:,atributos_continuos],0)
		self.minimos = np.min(datosTrain[:,atributos_continuos],0)
		self.A = (self.maximos - self.minimos)/self.K
		self.poblacion = self.generar_poblacion(len(diccionario[-1]))

		self.fitness_poblacion = []
		for i in range(self.tamano_poblacion):
			sd=self.fitness(datosTrain,self.poblacion[i])
			self.fitness_poblacion.append(sd) 

		while(self.generaciones > 0 and max(self.fitness_poblacion)<self.max_fitness and self.tiempo_sin_mejora < 25):
			ruleta = []
			#print("seleccion")
			self.poblacion=self.seleccion_progenitores(datosTrain)

			#print(self.fitness(datosTrain, self.poblacion[0]))
			#print("cruce")

			#for i in range(self.tamano_poblacion):
			#	self.fitness_poblacion[i] = self.fitness(datosTrain, self.poblacion[i])

			#probabilidades = np.divide(self.fitness_poblacion, np.sum(self.fitness_poblacion))
			aux = []
			posiciones = np.random.permutation(range(0,self.tamano_poblacion-len(self.elitismo)))
			for i in range(int(len(posiciones)/2)):
				
				elem1 = copy.deepcopy(self.poblacion[posiciones[i]])
				elem2 = copy.deepcopy(self.poblacion[posiciones[i+1]])
				aux.extend(self.Cruce(elem1,elem2))
				i +=1
			self.poblacion = aux
			#print(len(self.poblacion),self.tamano_poblacion-len(self.elitismo))
			for i in range(len(self.poblacion)):
				self.poblacion[i] = self.Mutacion(self.poblacion[i])

			self.poblacion.extend(self.elitismo)
			#print(len(self.poblacion))
			for i in range(self.tamano_poblacion):
				self.fitness_poblacion[i] = self.fitness(datosTrain, self.poblacion[i])



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

			self.generaciones -= 1

		print("")
		n_ganador = np.argsort(self.fitness_poblacion)[-1]
		self.regla = self.poblacion[n_ganador]
		print(self.regla)
		#print(self.fitness_poblacion[n_ganador])
		#print(self.fitness(datosTrain,self.regla))
		return

		
	def clasifica(self,datostest,atributosDiscretos,diccionario):
		num_reglas = len(self.regla)
		clasificacion = []
		#aciertos = 0;
		for dato in datostest:
			#Hallo el intevalo al que pertenece cada atributo y annado la clase al final
			salida = self.discretizar_elemento(dato)
			clase = []
			for regla in self.regla:
				flag_atributos = True
				for atributo in range(self.natributos):
					opciones = np.array(np.where(np.asarray(regla[atributo]) == 1)).ravel()
					if list(opciones)==[]:
						continue
					if not np.any(opciones == salida[atributo]):
						flag_atributos = False
						break
				if flag_atributos == True:
					clase.append(regla[-1])
			if len(clase) == 0:
				clase.append(self.priori)
			clasificacion.append(mode(clase)[0][0])
		return np.array(clasificacion)


	def Cruce(self,elem1, elem2):
		if(np.random.rand(1)[0] < self.probabilidad_recombinacion or (len(elem1)==1 and len(elem2)==1)):
			return [elem1,elem2]

		n = np.random.randint(1,max(len(elem1),len(elem2)))
		hijo1 = elem1[0:n]+elem2[n:]
		hijo2 = elem2[0:n]+elem1[n:]

		return hijo1, hijo2
		

	def Mutacion(self,elem):
		flag = False
		for i in range(len(elem)):
			for j in range(len(elem[i])-1):
				for k in range(len(elem[i][j])):
					if np.random.rand(1)[0]<self.probabilidad_mutacion:
						flag = True
						if elem[i][j][k]==1:
							elem[i][j][k] =0
						else:
							elem[i][j][k] = 1
		if len(elem) >2:	
			if np.random.rand(1)[0]<self.probabilidad_mutacion:
				elem.pop(np.random.randint(0,len(elem),1)[0])
		if len(elem) <20:
			if np.random.rand(1)[0]<self.probabilidad_mutacion:
				regla = []
				for j in range(self.natributos):
					rangos = 2**(self.K-1)-1
					valoresAtrib = np.random.permutation(np.asarray([int(d) for d in np.binary_repr(rangos, width=self.K)]))
					regla.append(valoresAtrib)
				#Le annadimos una clase aleatoria y metemos la regla en nuestro individuo
				regla.append(np.random.randint(0,2))
				elem.append(regla)
		return elem

	def discretizar_elemento(self,elem):
		columnas_interes = np.flatnonzero(elem[:-1]) #Todas las columnas menos la clase
		discretizado = np.zeros(len(elem))
		for c in columnas_interes:
			valor = np.ceil((elem[c] - self.minimos[c])/self.A[c])
			discretizado[c] = valor
		discretizado[-1] = elem[-1]
		return discretizado.astype(int)


	def seleccion_progenitores(self,datosTrain):
		aux = []
		aux2=0
		poblacion = []

		elementos_mantener = int(-1* self.proporcion_elitismo* len(self.poblacion))
		##print(elementos_mantener)
		posiciones = np.argsort(self.fitness_poblacion)
		self.elitismo = []
		for pos in posiciones[elementos_mantener:]:
			self.elitismo.append(copy.deepcopy(self.poblacion[pos]))

		probabilidades = np.divide(self.fitness_poblacion,np.sum(self.fitness_poblacion))

		#print(self.fitness_poblacion)
		#print(probabilidades)

		for i in range(len(probabilidades)):
			for aa in range(int(np.floor(probabilidades[i]*100))):
				aux.append(self.poblacion[i])



		for w in range(len(self.poblacion)-len(posiciones[elementos_mantener:])):
			aux2 = np.random.randint(0,len(aux))
			poblacion.append(copy.deepcopy(aux[aux2]))



		return poblacion