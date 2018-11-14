from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from plotModel import plotModel 
import matplotlib.pyplot as plt 

datasets = ["example1.data","example2.data","example3.data","example4.data","wdbc.data"]
n_particiones = 1
for i in range(len(datasets)):
    dataset = Datos(datasets[i])

    particiones = ValidacionSimple(n_particiones,0.5,dataset).creaParticiones(dataset.datos)   
    errores = np.empty(n_particiones)
    for j in range(n_particiones):
        datosTrain = dataset.extraeDatos(particiones[j].indicesTrain)
        datosTest = dataset.extraeDatos(particiones[j].indicesTest)
        cKNN =  ClasificadorVecinosProximos()
        cKNN.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
        #res = cKNN.clasifica(datosTest[:,0:-1],dataset.nominalAtributos,dataset.diccionarios)

        #error = Clasificador.error(datosTest,res)
        #errores[j] = error
        #"""
        plt.figure()
        plotModel(dataset.datos[particiones[j].indicesTrain,0],dataset.datos[particiones[j].indicesTrain,1],
                dataset.datos[particiones[j].indicesTrain,-1] !=0, cKNN,"Frontera",dataset.diccionarios)
        plt.plot(dataset.datos[dataset.datos[:,-1]==0,0],  
         dataset.datos[dataset.datos[:,-1]==0,1],'ro')
        plt.plot(dataset.datos[dataset.datos[:,-1]==1,0],  
         dataset.datos[dataset.datos[:,-1]==1,1],'bo')
        plt.show()
        #"""
    print(np.mean(errores))


