from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from sklearn.neighbors import KNeighborsClassifier

datasets = ["example1.data","example2.data","example3.data","example4.data","wdbc.data"]
n_particiones = 1
for i in range(len(datasets)):
    dataset = Datos(datasets[i])

    particiones = ValidacionSimple(n_particiones,0.1,dataset).creaParticiones(dataset.datos)   
    errores = np.empty(n_particiones)
    for j in range(n_particiones):
        datosTrain = dataset.extraeDatos(particiones[j].indicesTrain)
        datosTest = dataset.extraeDatos(particiones[j].indicesTest)
        cKNN =  ClasificadorVecinosProximos()
        cKNN.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios)
        res = cKNN.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios)

        error = Clasificador.error(datosTest,res)
        errores[j] = error
    print("Error del clasificador:"),
    print(np.mean(errores))

    neigh = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2) #pesos uniformes, norma 2, 5 vecinos
    neigh.fit(datosTrain[:,:-1],datosTrain[:,-1])
    res = neigh.predict(datosTest[:,:-1])
    error = Clasificador.error(datosTest,res)
    print("Error de scikit:"),
    print(error)
