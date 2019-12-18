from tools.tools import load_image, save_image
import torch


# Exercice 1
# La fonction load_image vous permet de charger une image depuis un fichier et
# en fait un tensor torch. Par exemple vous pouvez charger la photo d'isha avec
# x = load_image('isha.jpg').
#
# Testez cette fonction dans la section __main__ plus bas et regardez la forme
# du tensor x avec x.size()

# Exercice 2
# Le tenseur à le format suivant : Hauteur x Largeur x 3. Une image a en effet
# 3 cannaux : Rouge, Vert et Bleu.
# Vous pouvez accedez à chaque cannal de façon individuelle avec x[:, :, index_cannal].
# Utilisez la fonction save_image pour sauvegarder séparement les cannaux R, V, B
# de x.


# Exercice 3
# Les couleurs c'est bien jolis mais ce n'est pas pratique pour faire des manipulations.
# Definissez une fonction qui étant donné un vecteur image en fait une image
# en niveau de gris en moyennant les cannaux Rouges, Vert, Bleu

# L'image de sortie doit avoir le format suivant Hauteur x Largeur

def rgb_to_grey(x):
    # Votre code ici
    return

# Exercice 4
# Appliquons des filtres à des images: le plus souvent on appelle "filtre" une
# convolution appliquée à l'image. Pytorch possède un module codant les
# convolutions, il s'agit de nn.Conv2d.

# Par exemple la fonction suivante calcul le gradient selon l'axe x (grad_x)
def build_conv_grad_x():

    output = torch.nn.Conv2d(1,         # Nombre de cannaux dans l'image d'entrée, nous travaillons avec des images en noir et blanc donc 1
                             1,         # Nombre de cannaux dans l'image de sortie
                             3,         # Taille du noyau de comvolution
                             padding=1) # Nombre de lignes zéros à ajouter à l'image lors de la convolution

    # Le noyau de convolution est de taille trois, cela veut dire que la convolution est effectuée par une matrice 3x3
    core = torch.tensor([[0, 0, 0],
                         [-1, 0, 1],
                         [0, 0, 0]], dtype=torch.float)
    output.weight.data.copy_(core)
    return output

# La fonction ci dessous applique un filtre à une image noir et blanc au
# format Hauteur x Largeur
def applique_filtre(x, filtre):

    H, L = x.size()

    # Les modules torch.nn.Conv2d ne prennent en entrée que des batches au format
    # Nombres d'images x Nombre de cannals par image x Hauteur d'une image x Largeur d'une image
    # La méthode torch.tensor.view permet de faire ça: une image H x L c'est aussi
    # un batch ne contenant qu'une image avec un seul cannal
    x = x.view(1, 1, H, L)

    # Appliquer le filtre
    x = filtre(x)

    # Remettre x au format H, L
    x = x.view(H, L)

    # Retourner le résultat
    return x

# Utilisez les deux fonctions ci-dessus pour tester grad_x et utilisez save_image
# pour visualiser le résultat.
# Vous remarquerez que le gradient peut être négatif: puisque seule la valeur
# du gradient nous intéresse et non sa direction, utilisez la méthode torch.abs
# pour ne garder que la valeur absolue de grad_x

# Exercice 5
# De la même manière, définissez et testez:
# - une convolution pour extraire le gradient selon l'axe y
# - une convolution pour extraitre la somme des gradients selon les axes x et y
# - un filtre moyen: chaque pixel est remplacé par la moyenne de ses voisins sur
# un carré de taille 3x3
# - un gros filtre moyen: chaque pixel est remplacé par la moyenne de ses voisins sur
# un carré de taille 9x9 (Attention au padding !!!!)

def build_conv_grad_y():
    # Votre code ici
    return

def build_conv_grad_sum_xy():
    # Votre code ici
    return

def build_conv_mean_3x3():
    # Votre code ici
    return

def build_conv_mean_9x9():
    # Votre code ici
    return

# Exercice 6
# Regardons à présent ce qui se passe lorsque l'on combine des filtres.
# Regardez les effets des fonctions suivantes

def mean_3x3_combo(x, n_combo):
    mean_3x3 = build_conv_mean_3x3()
    for _ in range(n_combo):
        x = applique_filtre(x, mean_3x3)
    return x


def diff_mean_combo(x):
    mean_3x3 = build_conv_mean_3x3()
    return 2 * torch.abs(x - applique_filtre(x, mean_3x3))

# Definissez la comvoliution à laquelle correspond diff_mean_combo() :
def diff_mean_combo_2(x):
    return

# Definissez la comvoliution à laquelle correspond mean_3x3_combo(x, 2) :
def mean_3x3_combo_x2(x):
    return

# Exercice 7:
# Vous allez à présent programmer un détecteur de bord. Appliquez votre filtre
# build_conv_grad_sum_xy à une image I pour estimer son gradient grad_I.
# Construisez alors une image J valant 0 sur tous les points où abs(grad_I) < 100
# et 1 ailleurs.
# Faites variez ce seuil pour en regarder les effets

def extraction_des_bords(x):
    return

# Exercice 8
#

if __name__ == "__main__":

    x = load_image('isha.jpg')
    # A vous de jouez !

    # N'oubliez pas de définir rgb_to_grey
    x = rgb_to_grey(x)

    save_image(x, 'isha2.jpg')
