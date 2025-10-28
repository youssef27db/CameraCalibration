import numpy as np
import cv2
import glob


class PairwiseCalibrator:
    def __init__(self, chessboard_size, criteria):
        self.chessboard_size = chessboard_size
        self.criteria = criteria
        self.objpoints = []  # 3D Punkte in der realen Welt
        self.imgpoints = []  # 2D Punkte in der Bildebene
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.imgs = []

    def processImages(self, images):
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Finde die Ecken des Schachbrettmusters
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                self.objpoints.append(self.objp)

                # Verfeinere die Eckpunkte für genauere Ergebnisse
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)

                # Zeichne und zeige die gefundenen Ecken
                img = cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                self.imgs.append(img)
                
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

    def pairwiseCalibration(self):
        #TODO: Implement pairwise calibration logic
        
        pass