# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_openml
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.neural_network import MLPClassifier
#
# # Fetch the MNIST dataset
# mnist = fetch_openml('mnist_784', version=1)
#
# # Split the data and target
# x, y = mnist['data'], mnist['target']
#
# # Display an example digit
# some_digit = x.iloc[15000]
# some_digit_image = some_digit.values.reshape(28,28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
#
# # Split the dataset into training and test sets
# x_train, y_train, x_test, y_test = x[:60000], y[:60000], x[60000:], y[60000:]
#
# # Shuffle the training set
# shuffle_index = np.random.permutation(60000)
# x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
# x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
#
# # Create binary target vectors for the digit '5'
# y_train_5 = (y_train == 5)
# y_test_5 = (y_test == 5)
#
# # Create and train the MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(20,20))
# mlp.fit(x_train, y_train_5)
#
# # Make predictions
# predictions = mlp.predict(x_test)
#
# # Get all unique labels
# all_labels = np.unique(np.concatenate((y_test_5, predictions)))
#
# # Print confusion matrix and classification report
# print(confusion_matrix(y_test_5, predictions, labels=all_labels))
# print(classification_report(y_test_5, predictions))
# ----------------------------------------------------------------
#autre solution
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# Récupérer le jeu de données MNIST
mnist = fetch_openml('mnist_784', version=1)

# Diviser les données et les cibles
x, y = mnist['data'], mnist['target']

# Afficher un exemple de chiffre
some_digit = x.iloc[15000]
some_digit_image = some_digit.values.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")

# Diviser le jeu de données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = x[:60000].to_numpy(), x[60000:].to_numpy(), y[:60000], y[60000:]

# Mélanger l'ensemble d'entraînement
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# Créer des vecteurs cibles binaires pour le chiffre '5'
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# Créer et entraîner le MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,20))
mlp.fit(x_train, y_train_5)

# Faire des prédictions
predictions = mlp.predict(x_test)

# Afficher la matrice de confusion et le rapport de classification
print(confusion_matrix(y_test_5, predictions, labels=[True, False]))
print(classification_report(y_test_5, predictions))