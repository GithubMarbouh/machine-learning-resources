import numpy as np
def sigmoid(x):
# '''
# Sigmoid :  normalise les entrées
# @param x: (float)
# @return: float
# '''
 return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
# '''
# Calcule la dérivée de sigmoid
# @param x: (float) l'entrée est déjà une image par sigmoid @return: (float) la dérivée
# '''
 return x * (1 - x)
lamda = 1
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])
training_outputs = np.array([[0, 1, 1, 0]]).T
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1
print('Poids synaptiques aléatoires: ')
print(synaptic_weights)
i=0
for iteration in range(4):
   i+=1
   input_layer = training_inputs
   print(f"Entrées après l'entrainement {i} : ",input_layer)
   outputs = sigmoid(np.dot(input_layer, synaptic_weights))
   print(f"Sorties après l'entrainement {i} : ",outputs)
   # error1 = (training_outputs - outputs)**2
   error = (training_outputs - outputs)
   print(f"Erreur après l'entrainement {i} : ",error)
   # print("Erreur : ",error1)
   adjustments =2*lamda* error * sigmoid_derivative(outputs)
   print(f"Ajustements {i}  : ",adjustments)
   print(input_layer.T)
   synaptic_weights += np.dot(input_layer.T, adjustments)
   print(f"Poids synpatiques après l'entrainement {i} : ",synaptic_weights)
   print(f"--------------------- FIN ITERATION {i} --------------------------------")

print("Poids synpatiques après l'entrainement : ")
print(synaptic_weights)
print("Sorties après l'entrainement")
print(outputs)