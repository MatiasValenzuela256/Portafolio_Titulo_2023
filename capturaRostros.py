import cv2
import os
import imutils

personName = 'Matias' 
dataPath = 'D:\\Reconocimiento facial\\Data\\Fotos personas' 

personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)


capturadora = cv2.VideoCapture("rtsp://admin:admin1978@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")

while True:
    ret, frame = capturadora.read()
    cv2.imshow('captura', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = capturadora.read() 
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)
        count =count + 1
    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(1)
    if k == 27 or count >= 600:
        break
capturadora.release()
cv2.destroyAllWindows()