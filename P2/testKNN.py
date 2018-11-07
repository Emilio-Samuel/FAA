from Datos import *
from Clasificador import *
from EstrategiaParticionado import *

dataset = Datos("example1.data")
particiones1 = ValidacionSimple(1,0.3,dataset).creaParticiones(dataset.datos)   

for particiones in particiones1:
    datosTrain = dataset.extraeDatos(particiones.indicesTrain)
    datosTest = dataset.extraeDatos(particiones.indicesTest)
    cKNN =  ClasificadorVecinosProximos()
    cKNN.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
    res = cKNN.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios)

    error = Clasificador.error(datosTest,res)

    print(error)