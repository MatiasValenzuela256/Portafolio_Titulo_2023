import cv2 
import os 

dataPath = 'D:\\Reconocimiento facial\\Data\\Fotos personas' 
imagenPaths = os.listdir(dataPath)
print('imagenPaths=',imagenPaths)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Leer el modelo .xml creado anteriormente
face_recognizer.read('D:\\Reconocimiento facial\\Data\\Modelo personas\\modeloEigenFace.xml')

capturadora = cv2.VideoCapture("rtsp://admin:admin1978@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")

while True:
    ret, frame = capturadora.read()
    cv2.imshow('captura', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret,frame = capturadora.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        #EigenFaces
        if result[1] < 5000:
            cv2.putText(frame, '{}'.format(imagenPaths[result [0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame, 'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

capturadora.release()
cv2.destroyAllWindows()

