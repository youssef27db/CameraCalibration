import numpy as np
import cv2
import glob

# Definiere die Anzahl der inneren Ecken in der Schachbrettmuster
chessboard_size = (6, 8)
# abbruchkriterien für die Kalibrierung
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays zur Speicherung der Objektpunkte und Bildpunkte
objpoints = []  # 3D Punkte in der realen Welt
imgpoints = []  # 2D Punkte in der Bildebene

# Erstelle die Objektpunkte basierend auf der Schachbrettgröße
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Lade alle Bilder aus dem Verzeichnis 'images' mit der Endung .png
images = glob.glob('images/*.png')
index = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Finde die Ecken des Schachbrettmusters
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)

        # Verfeinere die Eckpunkte für genauere Ergebnisse
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Zeichne und zeige die gefundenen Ecken
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        
        # Zeige das Bild und warte auf Tasteneingabe (ESC zum Beenden, Leertaste für nächstes Bild)
        cv2.imshow('img', img)
        k = cv2.waitKey(0)

        if k == 27:
            break
        elif k == 32:
            index += 1
            if index >= len(images):
                print("Keine weiteren Bilder.")
                index = 0

cv2.destroyAllWindows()

# Kalibriere die Kamera mit den gesammelten Punkten
ret, mtx, dist, rotation_vecs, translation_vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Kameramatrix:\n", mtx)
print("\nVerzerrungskoeffizienten:\n", dist)

# Berechne den Reprojektionsfehler
total_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rotation_vecs[i], translation_vecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print("\nMittlerer Reprojektionsfehler: ", total_error / len(objpoints))

# Verzerre bild 0
img = cv2.imread('images/img0.png')
h, w = img.shape[:2]

distortion_strength = 25
dist_distort = dist * distortion_strength

# Erzeuge Verzerrungs-Mapping (inverse Richtung)
mapx, mapy = cv2.initUndistortRectifyMap(mtx, -dist_distort, None, mtx, (w, h), cv2.CV_32FC1)

# Wende das Mapping auf dein Bild an → ergibt künstlich verzerrtes Bild
distorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


cv2.imshow("Original", img)
cv2.imshow("Verzerrt", distorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Entzerre das verzerrte Bild
img = cv2.imread('images/img0.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Entzerre das Bild
undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
 
# Zuschneiden des Bildes basierend auf dem ROI
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

cv2.imshow('Entzerrt', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()