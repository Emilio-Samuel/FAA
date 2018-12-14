import numpy as np
from Clasificador import *
#SOLO 1 PARTICION!
class ClasificadorAGB(Clasificador):
	def __init__(self,tamano_poblacion = 50,probabilidad_recombinacion=0.4,probabilidad_mutacion=0.8,proporcion_elitismo=0.01,generaciones=10):

		self.tamano_poblacion = tamano_poblacion
		self.proporcion_elitismo = proporcion_elitismo
		self.probabilidad_mutacion = probabilidad_mutacion
		self.probabilidad_recombinacion = probabilidad_recombinacion
		#self.max_fitness = max_fitness
		self.generaciones = generaciones

	def generar_poblacion(self,nhip):
		#Inicializamos una poblacion vacia
		poblacion=[]
		aux = []		#Creamos un individuo
		for cromosoma in range(self.tamano_poblacion):
			individuo = []
			cabezon = []
			#Inicialmente un individuo tiene entre 1 y 10 reglas
			num_reglas = np.random.randint(1,20)
			for i in range(num_reglas):
				regla = []
				for j in range(self.natributos):
					#Para cada atributo de la regla creamos un array binario
					n= np.random.randint(0,3)
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
		
		#Para cada dato de entrenamiento
		for dato in datosTrain:
			#Hallo el intevalo al que pertenece cada atributo y annado la clase al final
			salida = self.discretizar_elemento(dato)
			for regla in elem:
				flag = True
				for atributo in range(self.natributos):
					opciones = np.array(np.where(np.asarray(regla[atributo]) == 1)).ravel()
					if list(opciones)==[]:
						continue
					if not np.any(opciones == salida[atributo]):
						flag = False
						break
				if flag == True and salida[-1]==regla[-1]:
					aciertos+=1
					break
				if flag == False and salida[-1]==0:
					aciertos+=1
					break

		return aciertos*1./len(datosTrain)








	def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):

		self.natributos = max(datosTrain.shape[1] -12,2)
		self.K = np.floor(1+ 3.322*np.log10(len(datosTrain)))
		self.K = self.K.astype(np.int64)
		print(self.K)
		atributos_continuos = [e for  e,x in enumerate(1- np.array(atributosDiscretos)) if x == 1]
		self.maximos = np.max(datosTrain[:,atributos_continuos],0)
		self.minimos = np.min(datosTrain[:,atributos_continuos],0)
		self.A = (self.maximos - self.minimos)/self.K
		self.poblacion = self.generar_poblacion(len(diccionario[-1]))

		self.fitness_poblacion = []
		for i in range(self.tamano_poblacion):
			sd=self.fitness(datosTrain,self.poblacion[i])
			print(sd)
			self.fitness_poblacion.append(sd) 

		#print(self.fitness_poblacion)

		while(self.generaciones > 0 ):#and max(self.fitness_poblacion)<self.max_fitness):
			print("seleccion")
			self.poblacion=self.seleccion_progenitores()
			print("cruce")
			for i in range(self.tamano_poblacion - 1):
				#print(self.poblacion[i])
				#print(self.poblacion[i])
				uno = np.random.randint(0,self.tamano_poblacion)
				dos = np.random.randint(0,self.tamano_poblacion)
				[self.poblacion[uno], self.poblacion[dos]] = self.Cruce(self.poblacion[i], self.poblacion[i+1])
				i += 1
			print("mutacion")
			for i in range(self.tamano_poblacion):
				self.poblacion[i] = self.Mutacion(self.poblacion[i])

			for i in range(self.tamano_poblacion):
				self.fitness_poblacion[i] = self.fitness(datosTrain,self.poblacion[i])

			self.generaciones -= 1

		n_ganador = np.argsort(self.fitness_poblacion)[-1]
		self.regla = self.poblacion[n_ganador]
		print(self.fitness_poblacion[n_ganador])
		print(self.fitness(datosTrain,self.regla))
		return

			
	def clasifica(self,datostest,atributosDiscretos,diccionario):
		return self.fitness(datostest,self.regla)





	def Cruce(self,elem1, elem2):
		if(np.random.rand(1)[0] > self.probabilidad_recombinacion or (len(elem1)==1 and len(elem2)==1)):
			return [elem1,elem2]

		n = np.random.randint(1,max(len(elem1),len(elem2)))
		'''for regla in range(min(len(elem1),len(elem2))):
			for atrib in range(self.natributos):
				i = np.random.randint(0,self.K)
				aux = elem2[regla][atrib]
				ev=elem1[regla][atrib][0:i]
				asa=elem2[regla][atrib][i:]
				elem1[regla][atrib] = np.concatenate((elem2[regla][atrib][0:i],elem1[regla][atrib][i:]))

				elem2[regla][atrib] = np.concatenate((aux[0:i],elem2[regla][atrib][i:]))'''
		return elem1[0:n]+elem2[n:], elem2[0:n]+elem1[n:]
		

	def Mutacion(self,elem):
		mutar = np.random.randint(0,1)
		if mutar > self.probabilidad_mutacion:
			self.elem[np.random.randint(0,len(elem))].reverse()

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
		aux = []
		aux2=0
		poblacion = []
		'''
		elementos_mantener = int(np.floor(-1* self.proporcion_elitismo* len(self.poblacion)))
		#print(elementos_mantener)
		posiciones = np.argsort(self.fitness_poblacion)
		
		for pos in posiciones:
			poblacion.append(self.poblacion[pos])
'''

		probabilidades = np.divide(self.fitness_poblacion,np.sum(self.fitness_poblacion))
		print(self.fitness_poblacion)

		for i in range(len(probabilidades)):
			aux.append([self.poblacion]*int(np.floor(probabilidades[i]*100)))

		for w in range(len(self.poblacion)):
			aux2 = np.random.randint(0,len(aux))
			poblacion.append(self.poblacion[aux2])

		return poblacion