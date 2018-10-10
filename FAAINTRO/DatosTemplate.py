import numpy as np

class Datos(object):
  
  TiposDeAtributos=('Continuo','Nominal')

 
  # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y diccionarios
  def __init__(self, nombreFichero):
   self.nominalAtributos = []
   self.tipoAtributo = []
   self.diccionarios = []
   valores = []
   listasDatos = []
   f = open(nombreFichero, "r")
   lineas = f.readlines()
   #numInstances =  int(lineas[0])
   nombreAtributos = lineas[1].rstrip("\r\n").split(",")
   self.tipoAtributos = lineas[2].rstrip("\r\n").split(",")
   print(self.tipoAtributos)
   for tipoAtributo in self.tipoAtributos:
      print(tipoAtributo)
      if tipoAtributo == self.TiposDeAtributos[1]:
         self.nominalAtributos.append(True)
         self.nominalAtributos.append(self.TiposDeAtributos[1])
      elif tipoAtributo == self.TiposDeAtributos[0]:
         self.nominalAtributos.append(False)
         self.nominalAtributos.append(self.TiposDeAtributos[1])
      else:
         raise ValueError("El atributo "+tipoAtributo+" no es ni nominal ni continuo")
      self.diccionarios.append({})
      valores.append([])
      listasDatos.append([])
   a = int(lineas[0].rstrip("\r\n"))
   b = len(nombreAtributos)
   print(a,b)
   datos = np.ones((a, b))

   for line in lineas[3:]:
      tupla = line.rstrip("\n").split(",")
      for i in range(0,len(tupla)):
         if tupla[i] not in valores[i] and self.nominalAtributos[i] == True:
            valores.append(tupla[i])
         listasDatos[i].append(tupla[i])
   valores.sort()
   print(listasDatos)
   for atributo in listasDatos:
      for i in range(0,len(atributo)):
         if self.nominalAtributos[i] == True:
            atributo[i] = valores.index(atributo[i])

   for i in int(lineas[0].rstrip("\r\n")):
      datos[:,i]=listasDatos[i]

  def extraeDatos(idx):
    pass


  
