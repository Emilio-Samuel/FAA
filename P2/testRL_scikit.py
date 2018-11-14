from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
from sklearn.linear_model import LogisticRegression

datasets = ["example1.data","example2.data","example3.data","example4.data","wdbc.data"]
n_particiones = 1
for i in range(len(datasets)):
    dataset = Datos(datasets[i])

    particiones = ValidacionSimple(n_particiones,0.1,dataset).creaParticiones(dataset.datos)   

    datosTrain = dataset.extraeDatos(particiones[0].indicesTrain)
    datosTest = dataset.extraeDatos(particiones[0].indicesTest)
    cRLs = LogisticRegression(solver='lbfgs', max_iter=500) #solver por defecto, se puede usar sag para resulatdos mas parecidos
    cRLs.fit(datosTrain[:,:-1],datosTrain[:,-1])
    res = cRLs.predict(datosTest[:,:-1])
    error = Clasificador.error(datosTest,res)
    print("Error de scikit:"),
    print(error)
