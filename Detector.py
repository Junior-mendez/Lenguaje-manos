import cv2
import mediapipe as mp
import os

#Creamos la carpeta donde almacenaremos el entrenamiento
nombre='Letra_A'
direccion='C:/Developer/upn/Lenguaje/Entrenamiento'
# direccion=os.path.dirname()+'/Entrenamiento'
carpeta= direccion+'/'+nombre

if not os.path.exists(carpeta):
    print('Carpeta Creada:', carpeta)
    os.makedirs(carpeta)

#Asignacion de un contador para el nombre de las fotos
cont=0

#leemos la camara
cap =cv2.VideoCapture(0)

##Creamos un objeto que va almacenar la detección y el seguimiento de las manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()

#Método para dibujar las manos
dibujo= mp.solutions.drawing_utils

while(1):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = [] #En esta lista vamos a almacenar las coordenadas de los puntos 
    print(resultado.multi_hand_landmarks)
    if (resultado.multi_hand_landmarks):# Si hay algo en los resultados entramos al if
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):

                alto, ancho, c = frame.shape
                corx, cory = int(lm.x*ancho), int(lm.y*alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) !=0:
                pto_i1 = posiciones[4] #Dedos: 4
                pto_i2 = posiciones[20] #Dedos: 20
                pto_i3 = posiciones[12] #Dedos: 12
                pto_i4 = posiciones[0] #Dedos: 0
                pto_i5 = posiciones[9] #Punto central
                x1, y1 = (pto_i5[1]-100), (pto_i5[2]-100) #Obtenemos el punto inicial y las longitudes
                ancho, alto = (x1+200), (y1+200)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame,(x1,y1), (x2,y2), (0,255,0),3)
            dedos_reg = cv2.resize(dedos_reg, (200,200), interpolation = cv2.INTER_CUBIC) #  redimensionamos las fotos
            cv2.imwrite(carpeta + "/Dedos_{}.jpg".format(cont),dedos_reg)
            cont = cont + 1
    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont>=300:
        break
cap.release()
cv2.destroyAllWindows()

