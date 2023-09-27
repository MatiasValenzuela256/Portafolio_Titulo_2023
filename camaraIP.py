import cv2

capturadora = cv2.VideoCapture("rtsp://admin:admin1978@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")

while True:
    ret, frame = capturadora.read()
    cv2.imshow('captura', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capturadora.release()
cv2.destroyAllWindows()




