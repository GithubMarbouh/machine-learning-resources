#importations préalables
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
# Génération des données bidimensionnelles complètes
n_base=100
data1=np.random.randn(n_base,2) + [5,5]
data2=np.random.randn(n_base,2) + [3,2]
data3=np.random.randn(n_base,2) + [1,5]
data=np.concatenate((data1, data2, data3))
data.shape # vérification
# (300, 2)
# print(data)
# print(data.shape)
#On mélange les données np.random.shuffle(data)
#visualisation (optionnelle) des données générées
plt.plot(data[:, 0], data[:,1], 'r+')
plt.show()

# Suppression de certaines valeurs de la seconde variable (colonne)
#pour obtenir des données manquantes
n_samples=data.shape[0]
#définition du taux de lignes à valeurs manquantes
missing_rate = 0.3
n_missing_samples=int(np.floor(n_samples*missing_rate))
print("Nous allons supprimer {} valeurs".format(n_missing_samples))
#choix des lignes à valeurs manquantes
present = np.zeros(n_samples-n_missing_samples, dtype=bool)
missing=np.ones(n_missing_samples, dtype=bool)
missing_samples=np.concatenate((present, missing))
#On mélange le tableau des valeurs absentes
np.random.shuffle(missing_samples)
print(missing_samples)
# obtenir la matrice de données avec des valeurs manquantes: manque indiqué par NaN par exemple
#valeurs NAN dans la seconde colonne pour les lignes True dans missing_samples
data_missing=data.copy()
data_missing[np.where(missing_samples), 1]=np.nan
print(data_missing)
# Imputation par la moyenne :les NaN sont remplacés par la moyenne de la colonne
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data_imputed = imp.fit_transform(data_missing)
print(data_imputed)
#visualisation (optionnelle) des données générées
plt.scatter(data_imputed[:, 0], data_imputed[:,1], c=missing_samples, marker='+')
plt.show()

# calcul l erreur quadratique moyenne entre les données complètes et les données imputées
mean_squared_error(data[missing_samples,1], data_imputed[missing_samples,1])

# 2)impultation par le centre du groupe ,pour les données generees
# obtenir le tableau compose des seules observations completes
data_filtered = data[~missing_samples, :]
print(data_filtered.shape)

# application de la classification automatique sur les données complètes
kmeans = KMeans(n_clusters=3).fit(data_filtered)
 # affichage des centres des groupes
centres = kmeans.cluster_centers_
print("Centres : {}".format(centres))
plt.scatter(data_filtered[:, 0], data_filtered[:,1], marker='+', c=kmeans.labels_)
plt.scatter(centres[:, 0], centres[:,1], edgecolors='k',marker='*', s=300, label='Centres',c=range(len(centres)))
plt.legend()
plt.show()

# Pour chaque observation incompl`ete, nous allons d´eterminer le centre
# le plus proche de cette observation (`a l’aide de l’outil NearestCentroid de scikit-learn).
# L’indice du centre le plus proche sera d´etermin´e `a partir des valeurs non manquantes.

y=np.array([1,2,3])
ncPredictor = NearestCentroid()
# les centres calcules par la methode kmeans sont associeas aux 3 etiquettes des groupes
ncPredictor.fit(centres[:,0].reshape(-1,1), y)
nearest = ncPredictor.predict(data_missing[missing_samples,0].reshape(-1,1))
estimated=np.zeros(n_missing_samples)
indeces = range(n_missing_samples)
for i in indeces:
    estimated[i]=centres[nearest[i]-1,1]
data_imputed = data_missing.copy()
data_imputed[missing_samples,1]=estimated
mean_squared_error(data[missing_samples,1], data_imputed[missing_samples,1])

plt.scatter(data_imputed[:, 0], data_imputed[:,1], c=missing_samples, marker='+')
plt.scatter(centres[:, 0], centres[:,1], edgecolors='k',marker='*', s=300, label='Centres',c='r')
plt.legend()
plt.show()