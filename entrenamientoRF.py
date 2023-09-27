import cv2 
import os 
import numpy as np

dataPath = 'D:\\Reconocimiento facial\\Data\\Fotos personas' 
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print ('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        cv2.imshow('image' ,image)
        cv2.waitKey(10)
    label = label + 1

print('labels= ', labels)
print('Numero de etiquetas 0: ', np.count_nonzero(np.array(labels)==0)) # CONTAR LAS ETIQUETAS
print('Numero de etiquetas 1: ', np.count_nonzero(np.array(labels)==1))
#print('Numero de etiquetas 2: ', np.count_nonzero(np.array(labels)==2))

face_recognizer = cv2.face.EigenFaceRecognizer_create()
# Entrenando el reconocimiento facial
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Especificar la ruta completa de la carpeta donde deseas guardar el archivo XML
modeloPath = 'D:\\Reconocimiento facial\\Data\\Modelo personas'
if not os.path.exists(modeloPath):
    os.makedirs(modeloPath)

# Almacenar el modelo obtenido en la carpeta especificada
modeloFile = os.path.join(modeloPath, 'modeloEigenFace.xml')
face_recognizer.write(modeloFile)
print("Modelo Almacenado en:", modeloFile)
