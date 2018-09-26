import numpy as np
class Datos (object):
        TiposDeAtributos=('Continuo','Nominal')
        # TODO: procesar el fichero para asignar correctamente las variables 
        # tipoAtributos, nombreAtributos,nominalAtributos, datos y diccionarios
        def __init__(self,nombreFichero):
                self.nombreFichero = nombreFichero
                f = open(nombreFichero,'r')
                self.fichero = f.readlines()
                NAtributos = int(self.fichero[0])
                #quitamos el salto de linea y el retorno de carro si es necesario
                aux = self.fichero[1][:-1].rstrip("\r")
                self.nombreAtributos = aux.split(',')
                cols = len(self.nombreAtributos)
                aux = self.fichero[2][:-1].rstrip("\r")
                self.tipoAtributos = aux.split(',')
                
                #Si no es cotinuo o nominal algun tipo marcamos como error
                if any(x != 'Continuo' and x != 'Nominal' for x in self.tipoAtributos):
                        raise Exception('ValueError')

                self.nominalAtributos = [x == 'Nominal' for x in self.tipoAtributos]
                
                matriz = list()
                self.diccionarios = []
                aux = [list() for i in range(len(self.nombreAtributos))]
                for x in self.fichero[3:]:
                        y = x.rstrip("\r\n").split(',')
                        matriz.append(y)

                        for i in range(cols):
                                if (y[i] not in aux[i] and self.nominalAtributos[i]):
                                        aux[i].append(y[i])
                for x in aux:
                        self.diccionarios.append(dict(zip(sorted(x,key=str.lower),range(len(x)))))

                print(self.diccionarios)

                self.datos = np.ones((NAtributos,len(self.nombreAtributos)))
                for i in range(NAtributos):
                        for j in range(cols):
                                if self.nominalAtributos[j]:
                                        self.datos[i,j] = self.diccionarios[j].get(matriz[i][j])
                                else:
                                        self.datos[i,j] = matriz[i][j]
                print(self.datos)
        def extraeDatos(self, idx):
                pass