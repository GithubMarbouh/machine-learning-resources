from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Expérience de prédiction du cancer du sein
cancer = load_breast_cancer()
forme_donnees = cancer['data'].shape
forme_cibles = cancer['target'].shape
donnees = cancer['data']
cibles = cancer['target']
forme_donnees = donnees.shape
forme_cibles = cibles.shape
print(forme_donnees)
print(forme_cibles)

donnees_entrainement, donnees_test, cibles_entrainement, cibles_test = train_test_split(donnees, cibles)

normalisateur = StandardScaler()
# Adapter uniquement aux données d'entraînement
normalisateur.fit(donnees_entrainement)

# Appliquer maintenant les transformations aux données :
donnees_entrainement = normalisateur.transform(donnees_entrainement)
donnees_test = normalisateur.transform(donnees_test)

mlp = MLPClassifier(hidden_layer_sizes=(60,60,60,60), max_iter=1000)
mlp.fit(donnees_entrainement, cibles_entrainement)

cross_val_score(mlp, donnees_entrainement, cibles_entrainement, cv=5, scoring='accuracy')
predictions = mlp.predict(donnees_test)

print(confusion_matrix(cibles_test, predictions))

print(classification_report(cibles_test, predictions))