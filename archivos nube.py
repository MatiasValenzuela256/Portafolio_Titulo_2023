import boto3
import os
import cv2
import imutils


personName = 'Matias' 
carpeta_local = 'D:\\Reconocimiento facial\\Data\\Fotos personas' 

personPath = carpeta_local + '/' + personName

if not os.path.exists(personPath):
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)


capturadora = cv2.VideoCapture (0)#("rtsp://admin:admin1978@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")

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
    if k == 27 or count >= 50:
        break

     
# Configura las credenciales de AWS
aws_access_key_id = ''
aws_secret_access_key = ''

# Crea un cliente de S3
s3 = boto3.client('s3',
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key)

# Nombre del bucket de S3 donde deseas guardar los archivos
bucket_name = 'matiasivp1'

# Ruta local de la carpeta que contiene los archivos que deseas subir
carpeta_local = 'D:\\Reconocimiento facial\\Data\\Fotos personas'

# Prefijo para simular una carpeta en S3
prefijo_s3 = 'Fotos personas/Matias/'

# Recorre la carpeta y sube cada archivo al bucket de S3 con el prefijo
for root, dirs, files in os.walk(carpeta_local):
    for file in files:
        archivo_local = os.path.join(root, file)
        nombre_archivo_s3 = prefijo_s3 + os.path.relpath(archivo_local, carpeta_local)

        try:
            s3.upload_file(archivo_local, bucket_name, nombre_archivo_s3)
            print(f'El archivo {nombre_archivo_s3} se ha cargado correctamente en el bucket {bucket_name}.')
        except Exception as e:
            print(f'Ocurrió un error al cargar el archivo {nombre_archivo_s3} en S3: {str(e)}')

capturadora.release()
cv2.destroyAllWindows()