import numpy as np
from Clasificador import *
#SOLO 1 PARTICION!
class ClasificadorAGB(Clasificador):
	def __init__(self,tamano_poblacion = 50,probabilidad_recombinacion=0.1,probabilidad_mutacion=0.001,proporcion_elitismo=0.05,generaciones=1):

		self.tamano_poblacion = tamano_poblacion
		self.proporcion_elitismo = proporcion_elitismo
		self.probabilidad_mutacion = probabilidad_mutacion
		self.probabilidad_recombinacion = probabilidad_recombinacion
		#self.max_fitness = max_fitness
		self.generaciones = generaciones

	def generar_poblacion(self,nhip):
		poblacion=[]
		for cromosoma in range(self.tamano_poblacion):
			individuo = []
			num_reglas = np.random.randint(1,10)
			regla = []
			for i in range(num_reglas):
				for j in range(self.natributos):
					rangos=np.random.randint(0,2**(self.K+1)-1)
					valoresAtrib = np.asarray([int(d) for d in np.binary_repr(rangos, width=self.K+1)])
					regla.append(valoresAtrib)
				regla.append(np.random.randint(0,2))
				individuo.append(regla)
			poblacion.append(individuo)
		return poblacion


	def fitness(self,datosTrain,elem):
		aciertos = 0
		num_reglas = len(elem)
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
				for i in range(self.natributos):
					#Obtenemos los intervalos aceptados
					opciones = np.array(np.where(np.asarray(regla[i]) == 1)).ravel()
					#Si el ejemplo tiene la misma clase que nuestro cromosoma
					if salida[-1]==clase:
						#Y se cumple que algun atributo del ejemplo esta en el rango valido
						flag = np.any(opciones==salida[i])
					if flag:
						#Aumentamos su fitness y pasamos al siguiente ejemplo
						aciertos +=1
						flag = False
		return aciertos

	def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):

		self.natributos = datosTrain.shape[1] -1
		self.K = np.floor(1+ 3.322*np.log10(len(datosTrain)))
		self.K = self.K.astype(np.int64)
		atributos_continuos = [e for  e,x in enumerate(1- np.array(atributosDiscretos)) if x == 1]
		self.maximos = np.max(datosTrain[:,atributos_continuos],0)
		self.minimos = np.min(datosTrain[:,atributos_continuos],0)
		self.A = (self.maximos - self.minimos)/self.K
		self.poblacion = self.generar_poblacion(len(diccionario[-1]))
		self.fitness_poblacion = []
		for i in range(self.tamano_poblacion):
			self.fitness_poblacion.append(self.fitness(datosTrain,self.poblacion[i])) 
		#print(self.fitness_poblacion)

		while(self.generaciones > 0 ):#and max(self.fitness_poblacion)<self.max_fitness):
			print("seleccion")
			self.poblacion=self.seleccion_progenitores()
			print("cruce")
			for i in range(self.tamano_poblacion - 1):
				#print(self.poblacion[i])
				#print(self.poblacion[i])
				[self.poblacion[i], self.poblacion[i+1]] = self.Cruce(self.poblacion[i], self.poblacion[i+1])
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
		clasifica=0
		for dato in datostest:
			salida = self.discretizar_elemento(dato)
			for regla in self.regla:
				flag = True
				for i in range(self.natributos):
					if regla[-1] != salida[-1]:
						flag = False
						break
					#Obtenemos los intervalos aceptados
					opciones = np.array(np.where(np.asarray(regla[i]) == 1)).ravel()
					flag = np.any(opciones==salida[i])
					if flag == False:
						break
				if flag == True:
					clasifica += 1
					break
		return clasifica/len(datostest)





	def Cruce(self,elem1, elem2):
		if(np.random.rand(1)[0] > self.probabilidad_recombinacion or (len(elem1)==1 and len(elem2)==1)):
			return [elem1,elem2]

		n = np.random.randint(1,max(len(elem1),len(elem2)))

		return elem1[0:n]+elem2[n:], elem2[0:n]+elem1[n:]
		

	def Mutacion(self,elem):
		mutar = np.random.randint(0,1)
		if mutar > self.probabilidad_mutacion:
			self.elem.pop()

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
		poblacion = []
		probabilidades = np.cumsum(self.fitness_poblacion)/np.sum(self.fitness_poblacion)

		for i in range(len(probabilidades)):
			aux.append([self.poblacion]*int(np.floor(probabilidades[i]*100)))

		for a in range(len(self.poblacion)):
			poblacion.append(np.random.choice(self.poblacion))

		return poblacion