# Un petit programme illustrant la détection de bords.
# Pour cela nous allons utiliser l'algorithme Canny de la librairie OpenCV.
# Sur une image en noir et blanc I, Canny fonctionne de la manière suivante:
# 1) Calculer la norme du gradient sur l'image I
# 2) Garder tous les maximums locaux du gradient sur I
# 3) Appliquer un double seuil aux points retenus. Soit ||g_ij|| la norme du gradient
# du pixel (i,j)
#    - si ||g_ij|| > t1 alors (i,j) est un bord "fort"
#    - si t1 > ||g_ij|| > t2 alors (i,j) est un bord "faible"
#    - si t2 > ||g_ij|| alors (i,j) n'est pas un bord
# 4) Ne garder que les bords forts et les bords "faibles" connectés
# à un bord "fort"

# Importe la librairie Opencv avec ses librairies très utiles
import cv2

# Ouvre une nouvelle fenêtre
cv2.namedWindow("Canny")

# Capture la sortie de la webcam
vc = cv2.VideoCapture(0)

# Vérifie que la webcam transmet en effet des images
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

# Seuils de la norme gradient pour la détection de bords.
# Les pixels prennent des valeurs entières comprises entre 0 (noir)
# et 255 (blanc). Plus le seuil choisi est bas, plus grand seront le
# nombre de bords détectés.
threshold1 = 150
threshold2 = 125

# Fait tourner le programme en continu
while rval:
    # Convertit l'image en cours en noir et blanc
    black_and_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecte les bords avec Canny
    edges = cv2.Canny(black_and_white, threshold1, threshold2)

    # Affiche l'image dans la fenêtre "Canny"
    cv2.imshow("Canny", edges)

    # Capture la prochaine image de la webcam
    rval, frame = vc.read()

    # Si l'utilisateur appuie sur la touche numéro 20 (ESC) quitter la
    # boucle
    key = cv2.waitKey(20)
    if key == 27:
        break

# Détruit la fenêtre
cv2.destroyWindow("preview")
