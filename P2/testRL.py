from Datos import *
from Clasificador import *
from EstrategiaParticionado import *
datasets = ["example1.data","example2.data","example3.data","example4.data","wdbc.data"]
n_particiones = 20
for i in range(len(datasets)):
    dataset = Datos(datasets[i])

    particiones = ValidacionSimple(n_particiones,0.1,dataset).creaParticiones(dataset.datos)   
    errores = np.empty(n_particiones)
    for j in range(n_particiones):
        datosTrain = dataset.extraeDatos(particiones[j].indicesTrain)
        datosTest = dataset.extraeDatos(particiones[j].indicesTest)
        cRL =  ClasificadorRegresionLineal()
        cRL.entrenamiento(datosTrain,dataset.nominalAtributos,dataset.diccionarios,1,100)
        res = cRL.clasifica(datosTest,dataset.nominalAtributos,dataset.diccionarios)

        error = Clasificador.error(datosTest,res)
        errores[j] = error
    print(np.mean(errores))
