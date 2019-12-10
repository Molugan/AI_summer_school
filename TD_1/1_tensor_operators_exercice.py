import torch

# En python, les fonctions se définissent comme suit:
# def nom_de_la_fonction( arguments de la fonction):
#   commande 1
#   commande 2
#   return La valeur retournée par la fonction

# Par exemple, la fonction suivante fait la somme de x + y
def sum_x_y(x, y):
    return x + y


# Exercice 1: Définissez une fonction "norme" qui donne la norme L2 du
# tensor x

def norm(x):
    r"""
    Argument :
        x un tenseur de taille arbitraire
    Remvoie:
        sqrt(sum (x_i * x_i)) où x_i sont les éléments de x
    """
    # Votre code ici
    return

# Maintenant considérons une série de K vecteurs de dimension D.
# La manière standard de représenter ces données est par un tensor de dimension
# K x D. Définissez une fonction qui prend en entrée un tel tensor X et
# retourne un vecteur de dimension K, contenant la norme L2 des K "sous-vecteurs"
# de X.

def partial_norm(x):
    r"""
    Argument :
        X un tensor de taille K x D
    Renvoie:
        un tensor Nx de taille K tel que Nx[i] = sqrt(sum(X[i, j] * X[i, j]))
    """
    # Votre code ici
    return


# Exercice 2:
# Nous allons à présent définir un filtre très souvent utilisé en deep learning:
# instanceNorm. Cette fois-ci nous ne considérons plus des vecteurs mais des
# séquences de vecteurs. Nous avons donc N séquences, chacunes contenant S vecteurs
# de dimension D. On peut donc stocker ces données dans un tensor de dimension
# N x S x D  !
# La première dimension est appelée le batch, la deuxième est le temps, et la
# troisième le cannal.
# Nous voulons donc calculer pour chaque séquence x, le vecteur moyen mu (de dimension D)
# x sur l'ensemble de ses S vecteurs ainsi que sa variance sigma (de dimension D) sur chacune de
# ses dimension.

# Commencez par écrire les fonctions permettant de calculer mu et sigma

def get_mean_on_seq(batch):
    r"""
    Argument :
        batch : un tensor de taille N x S x D
    Renvoie:
        Un tensor mu de dimension N x D telle que mu[n, d] = moyenne_{s}(x[n, s, d])
    """
    # Votre code ici: x_mu
    return

def get_var_seq(batch):
    r"""
    Argument :
        batch : un tensor de taille N x S x D
    Renvoie:
        Un tensor sigma de dimension N x D telle que sigma[n, d] = variance_{s}(x[n, s, d])
    """
    # Votre code ici: x_mu
    return

# Nous voulons à présent que chaque séquence x ait une moyenne nulle et une variance 1
# en d'autres termes nous voulons calculer : x - mu / sqrt(epsilon + sigma)
# epsilon est une petite valeur positive pour éviter de diviser par zero.
# On peut prendre par exemple epsilon = 1e-8
# Pour éviter d'avoir une erreur, ajoutez le paramètre keepdim=True dans la
# methode sum()

def instance_norm(batch):
    r"""
    Argument :
        batch : un tensor de taille N x S x D
    Renvoie:
        Un tensor norm_batch de taille NxSxD tel que
        norm_batch[n, s, d] = batch[n,s,d] - mu[n, d] / sqrt(epsilon + sigma[n, d])
    """
    # Votre code ici
    return


# Exercice 3:
# Maintenant définissez batch_norm ! Il s'agit d'une méthode très similaire à
# instance_norm sauf que l'on moyenne non seulement dans le temps mais aussi
# sur l'ensemble des séquences du batch.
# Cette fois-ci, utilisez les méthodes .mean() et .var() disponibles avec
# pytorch. Elles s'utilisent de la même manière que .sum()

def batch_norm(x):
    r"""
    Argument :
        batch : un tensor de taille N x S x D
    Renvoie:
        Un tensor norm_batch de taille NxSxD tel que
        norm_batch[n, s, d] = batch[n,s,d] - mu[d] / sqrt(epsilon + sigma[d])
    """
    # Votre code ici
    return

# BONUS : K-MEANS
# L'algorithme de K-MEAN est une méthode assez connue pour partitionner des
# vecteurs https://fr.wikipedia.org/wiki/K-moyennes
# En avec des tensors pytorch, vous n'avez besoin que de 4 lignes
# pour écrire cet algorithme.

# Indices:
# 1) Vous aurez besoin de la methode .min() qui peut s'utiliser comme sum()
# 2) si x est de dimension NxS et y est de dimension N alors x[y>0] vous donnera
# toutes les séquences de xi (de dimension S) de telles que yi > 0

def kMEANS(x, k, n_iterations):
    r"""
    Argument :
        x : un tensor de taille N x S
        k : nombre de clusters à calculer
        n_iterations : nombre maximal d'iterations
    Renvoie:
        k clusters
    """
    # Votre code ici
    return


if __name__ == "__main__":

    # Tester vos fonctions en les appelant ici !
    # Par exemple
    x = torch.randn((2,2))
    y = torch.randn((2,2))
    print(sum_x_y(x, y))
    print(x.sum(dim=1, keepdim=True))
