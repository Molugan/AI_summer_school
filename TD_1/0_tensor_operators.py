# Dans un terminal python testez les quelques unes, commandes suivantes.
# Vous y trouverez les outils nécessaires pour faire le premier exercice.

# Commencez par importer torch
import torch

###############################################################################
# Dans pytorch, les données sont organisées sous forme de tensor. Un tenseur
# est un tableau (ou vecteur) de nombres.

### Regardez ces quelques exemples de construction de tensors ###
print("Exemple 1: un tensor vide de dimension 2x3")
x = torch.empty(2, 3)
print(x)


print("Exemple 2: un tensor de valeurs aléatoires")
x = torch.rand(3, 3)
print(x)

print("Exemple 3: un tensor de zéros")
x = torch.zeros(5, 3)
print(x)

print("Exemple 4: un tensor avec des valeurs")
x = torch.tensor([[1,2],[3,4]])
print(x)
print(x.size())


### Les tensors s'utilisent comme des vecteurs ###

print("Opérations sur des tensors")
x = torch.randn(2, 3)
y = torch.randn(2, 3)
print("x :")

print("y :")
print(y)


print("x + y:")
print(x+y)

print("x * y:")
print(x*y)

print("x * y:")
print(x*y)

print("2*x:")
print(2*x)

print("x + 1:")
print(x + 1)

### Vous pouvez bien sur acceder aux valeurs des tensors ###
print("x[0,1]")
print(x[0,1])

print("x[0]")
print(x[0])

print("x[:, 2]")
print(x[:, 2])

### Vous pouvez même aller plus loin et accédez aux valeurs d'un tenseurs respectant
# certaines conditions

print("x > 0")
print(x[x > 0])

print("y là où x > 0")
print(y[x > 0])

### Il existe aussi plein de méthodes sur les tenseur ###

# Somme: regarder les différences entre ces trois commandes
print("Sum")
print(x.sum()) # Somme tous les éléments de x
print(x.sum(dim=0)) # Somme uniquement sur la première dimension
print(x.sum(dim=1)) # Somme uniquement sur la deuxième dimension
print(x.sum(dim=1, keepdim=True)) # Somme uniquement sur la deuxième dimension
