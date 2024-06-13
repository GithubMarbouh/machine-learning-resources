import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
# import cv2
# generation de donnes selon une loi normale tridimensionnelle
rndn3d= np.random.randn(500, 3)
 # affichage de nuage de points
fig= plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.scatter(rndn3d[:,0], rndn3d[:,1], rndn3d[:,2])
plt.title('Donnes initiales')
plt.show()

# applicarion de l'analyse en composantes principales
pca= PCA(n_components=3)
pca.fit(rndn3d)
print ('Pourcentage de variance espliquee:')
print (pca.explained_variance_ratio_)
print ('Composantes principales:')
print (pca.components_)

# affichage des donnees projetees
rndn3d_proj= pca.transform(rndn3d)
fig= plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.scatter(rndn3d_proj[:,0], rndn3d_proj[:,1], rndn3d_proj[:,2])
plt.title('Donnes projetees')
# changer la couleur des points
ax.scatter(rndn3d_proj[:,0], rndn3d_proj[:,1], rndn3d_proj[:,2], c='r')
plt.show()

# Appliquez maintenant une d´eformation et une rotation dans l ?
# espace tridimensionnel au nuage des observations de rndn3d, visualisez le r´esultat :
s1 = np.array([[3,0,0],[0,1,0],[0,0,0.2]])  # matrice de déformation
r1 = np.array([[0.36,0.48,-0.8],[-0.8,0.6,0],[0.48,0.64,0.6]])  # matrice de rotation
rndef = rndn3d.dot(s1).dot(r1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rndef[:,0], rndef[:,1], rndef[:,2])
plt.title("Données déformées")
plt.show()

pca = PCA(n_components=3)
pca.fit(rndef)
print("Pourcentage de variance expliquée : ")
print(pca.explained_variance_ratio_)
print("Composantes principales : ")
print(pca.components_)

# G´en´erez un autre ensemble de donn´ees
# en utilisant les mˆemes matrices de transformation
# et appliquez l’ACP `a ces nouvelles donn´ees


# reprendre L'ACP avec 2 composantes principales
# pca= PCA(n_components=2)
# pca.fit(rndn3d)
# print ('Pourcentage de variance espliquee:')
# print (pca.explained_variance_ratio_)
# print ('Composantes principales:')
# print (pca.components_)
# # affichage des donnees projetees
# rndn3d_proj= pca.transform(rndn3d)
# fig= plt.figure()
# ax= fig.add_subplot(111)
# ax.scatter(rndn3d_proj[:,0], rndn3d_proj[:,1])
# plt.title('Donnes projetees')
# plt.show()
# # reprendre L'ACP avec 1 composante principale
# pca= PCA(n_components=1)
# pca.fit(rndn3d)
# print ('Pourcentage de variance espliquee:')
# print (pca.explained_variance_ratio_)
# print ('Composantes principales:')
# print (pca.components_)
# # affichage des donnees projetees
# rndn3d_proj= pca.transform(rndn3d)
# fig= plt.figure()
# ax= fig.add_subplot(111)
# ax.scatter(rndn3d_proj[:,0], np.zeros(500))
# plt.title('Donnes projetees')
# plt.show()
# ----------------------------------------------------------------------------------------------
# application de l'ACP sur une image
# Charger l'image
# img = Image.open("mon_chien_exemple.jpeg")
img = Image.open("oiseau-colore.jpg")
# img = Image.open("perroquet-colore.jpg")

# Convertir l'image en tableau numpy
img_array = np.array(img)

# Aplatir le tableau
flat_img_array = img_array.reshape(-1, 3)  # Supposant que l'image est en RGB
print(img_array.shape[0]*img_array.shape[1],' ----',flat_img_array.shape[0])
s1 = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 0.2]])  # matrice de déformation
r1 = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]])  # matrice de rotation
transformed_data = flat_img_array.dot(s1).dot(r1)

# Appliquer l'ACP
pca = PCA(n_components=3)  # Vous pouvez choisir le nombre de composantes ici
pca.fit(transformed_data)

# Réduire la dimensionnalité
reduced_img_array = pca.transform(transformed_data)

# Reconstruire l'image à partir des composantes principales sélectionnées
reconstructed_img_array = pca.inverse_transform(reduced_img_array).reshape(img_array.shape)

# Afficher l'image originale et l'image reconstruite
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Image originale")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img_array.astype(np.uint8))
plt.title("Image reconstruite avec ACP")
plt.show()