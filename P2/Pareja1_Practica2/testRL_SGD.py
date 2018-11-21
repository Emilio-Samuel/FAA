from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from sklearn.linear_model import SGDClassifier

datasets = ["example1.data","example2.data","example3.data","example4.data","wdbc.data"]
n_particiones = 1
for i in range(len(datasets)):
    dataset = Datos(datasets[i])

    particiones = ValidacionSimple(n_particiones,0.1,dataset).creaParticiones(dataset.datos)   
    datosTrain = dataset.extraeDatos(particiones[0].indicesTrain)
    datosTest = dataset.extraeDatos(particiones[0].indicesTest)
    sgd = SGDClassifier(loss='log', max_iter=500, learning_rate='optimal') #regresion logistica, con 500 iteraciones. No hay equivalente para nuestra tasa de aprendizaje
    sgd.fit(datosTrain[:,:-1],datosTrain[:,-1])
    res = sgd.predict(datosTest[:,:-1])
    error = Clasificador.error(datosTest,res)
    print("Error de scikit:"),
    print(error)
