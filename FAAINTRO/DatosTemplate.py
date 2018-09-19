import numpy as np

class Datos(object):
  
  TiposDeAtributos=('Continuo','Nominal')

 
  # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y diccionarios
  def __init__(self, nombreFichero):
  	self.nominalAtributos = []
  	self.tipoAtributo = []
  	f = open(nombreFichero, "r")
  	lineas = f.readlines()
  	#numInstances =  int(lineas[0])
  	nombreAtributos = lineas[1].split(",")
  	# Necesidad de crear una excepcion en caso que un atributo no sea nominal ni continuo
  	self.tipoAtributos = lineas[2].split(",")
  	for tipoAtributo in nombreAtributos:
  		if tipoAtributo == TiposDeAtributos[1]:
  			self.nominalAtributos.append(True)
  			self.nominalAtributos.append(TiposDeAtributos[1])
  		elif tipoAtributo == TiposDeAtributos[0]:
  			self.nominalAtributos.append(False)
  			self.nominalAtributos.append(TiposDeAtributos[1])
  		else:
  			raise ValueError()


    
  # TODO: implementar en la práctica 1
  def extraeDatos(idx):
    pass


  
