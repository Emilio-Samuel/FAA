import numpy as np
from Clasificador import *
#SOLO 1 PARTICION!
class ClasificadorAGB(Clasificador):
	def __init__(self,tamano_poblacion = 50,probabilidad_recombinacion=0.1,probabilidad_mutacion=0.001,proporcion_elitismo=0.05,generaciones=50, max_fitness=0.95):

		self.tamano_poblacion = tamano_poblacion
		self.proporcion_elitismo = proporcion_elitismo
		self.probabilidad_mutacion = probabilidad_mutacion
		self.probabilidad_recombinacion = probabilidad_recombinacion
		self.max_fitness = max_fitness
		self.generaciones = generaciones

	def generar_poblacion(self,nhip):
		poblacion=[]
		for cromosoma in range(self.tamano_poblacion):
			individuo = list(np.random.randint(0,self.natributos*2^(self.K+1)-1,self.natributos))
				for j in len(individuo):
					individuo[j]=np.asarray([int(d) for d in str(bin(individuo[j]))[2:]])
			poblacion.append(individuo)
		return poblacion

	def fitness(self,datosTrain,elem):
		aciertos = 0
		for dato in datosTrain:
			salida = self.discretizar_elemento(dato)
			flag = False
			for regla in elem:
				if((salida == regla).all() or ((salida[:-1] != regla[:-1]).any() and salida[-1] != regla[-1])):
					flag = True
					break
			if flag:
				aciertos +=1
				
		return aciertos * 1./len(datosTrain)


	def fitness(self,datosTrain,elem):
		aciertos = 0
		#Para cada dato de entrenamiento
		for dato in datosTrain:
			#Hallo el intevalo al que pertenece cada atributo y annado la clase al final
			salida = self.discretizar_elemento(dato)
			#Inicializo una flag que me dirá si mi cromosoma es valido para este ejemplo
			flag = False
			#Para cada regla de mi cromosoma
			for regla in elem:
				#Me guardo la clase
				clase = regla[-1]
				#Compruebo para cada atributo del ejemplo
				for atributo in salida
					#Obtenemos los intervalos aceptados
					opciones = np.array(np.where(regla == 1)).ravel()
					opciones = opciones +1
					#Si el ejemplo tiene la misma clase que nuestro cromosoma
					if salida[-1]==clase:
						#Y se cumple que algun atributo del ejemplo esta en el rango valido
						flag = np.any(opciones==atributo)
					if flag:
						#Aumentamos su fitness y pasamos al siguiente ejemplo
						aciertos +=1
						break
				if flag:
					break
			if flag:
				break	
		return aciertos * 1./len(datosTrain)

	def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):

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

		while(self.generaciones > 0 and np.max(self.fitness_poblacion) < self.max_fitness):
			self.poblacion = self.seleccion_progenitores()
			for i in range(self.tamano_poblacion - 1):
				[self.poblacion[i], self.poblacion[i+1]] = self.Cruce(self.poblacion[i], self.poblacion[i+1])
				i += 1
			for i in range(self.tamano_poblacion):
				self.poblacion[i] = self.Mutacion(self.poblacion[i])

			for i in range(self.tamano_poblacion):
				self.fitness_poblacion[i] = self.fitness(datosTrain,self.poblacion[i])

			self.generaciones -= 1

		n_ganador = np.argsort(self.fitness_poblacion)[-1]
		self.regla = self.poblacion[n_ganador,:]
		print(self.generaciones)
		print(self.regla)
		print(self.fitness(datosTrain,self.regla))
		return


	def clasifica(self,datostest,atributosDiscretos,diccionario):
		return 1 - self.fitness(datostest,self.regla)

	def Cruce(self,elem1, elem2):
		if(np.random.rand(1)[0] > self.probabilidad_recombinacion):
			return [elem1,elem2]

		if elem1.shape[0] == elem2.shape[0] and elem.shape[0] == 1:

			n = np.random.randint(1,len(elem1),1)[0]

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
		discretizado[-1] = elem[-1]
		return discretizado.astype(int)


	def seleccion_progenitores(self):
		elementos_mantener = int(np.floor(-1* self.proporcion_elitismo* len(self.poblacion)))
		#print(elementos_mantener)
		posiciones = np.argsort(self.fitness_poblacion)
		poblacion = np.copy(self.poblacion[posiciones[elementos_mantener:]])

		probabilidades = np.cumsum(self.fitness_poblacion)/np.sum(self.fitness_poblacion)
		probabilidades = np.hstack((0,probabilidades))
		decision = np.random.rand(len(self.poblacion) + elementos_mantener)

		for p in decision:
			#print(np.where(probabilidades < p)[0], p)
			elegido = np.where(probabilidades < p)[0][0]
			poblacion = np.vstack((poblacion,self.poblacion[elegido,:]))
		return poblacion