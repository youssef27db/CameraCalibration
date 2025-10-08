import cv2

# Erstelle ein VideoCapture Objekt für die Kamera (Index 0 = Standard-Kamera)
cap = cv2.VideoCapture(0)

# Zähler für die Bildnummerierung
num = 0

while cap.isOpened():
    success, img = cap.read()

    # Warte auf Tasteneingabe (1ms Timeout)
    k = cv2.waitKey(1)

    # Wenn 's' gedrückt wird, speichere das aktuelle Bild
    if k == ord('s'):
        cv2.imwrite(f'images/img{num}.png', img)
        print(f'images/img{num}.jpg saved!')
        num += 1
        
    # Wenn ESC-Taste (ASCII-Code 27) gedrückt wird, beende die Schleife
    elif k == 27:
        break

    # Zeige das aktuelle Kamerabild in einem Fenster namens 'Image' an
    cv2.imshow('Image', img)

# Gebe die Kamera-Ressource frei
cap.release()
# Schließe alle OpenCV-Fenster
cv2.destroyAllWindows()