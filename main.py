#Importamos las librerias
from ultralytics import  YOLO
import cv2

#Leer nuestro modelo
model = YOLO("molinosag.pt")

#Realizar videocaptura
cap = cv2.VideoCapture(1)

#Bucle
while True:
    #Leer nuestros fotogramas
    ret, frame = cap.read()

    #Leemos resultados
    resultados = model.predict(frame, imgsz = 640 , conf = 0.90)

    #Mostramos resultados
    anotaciones = resultados[0].plot()


    #MOstramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    #Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
