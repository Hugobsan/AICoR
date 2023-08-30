import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

# X = Dados de entrada
X = [ [1,1,1,1,0,1,0,0,1,0], [1,1,0,1,1,1,1,1,0,1] ]

yd = [1, 0] # 1 = T, 0 = H

ptn = Perceptron(max_iter=5000,eta0=0.01,tol=1)
ptn.fit(X, yd, None, None)
y_pred=ptn.predict(X)

print("yd = ",yd)
print("y = ",y_pred)
if (yd == y_pred).all():
    print('O perceptron acertou todos os dados de treinamento')
else:
    print('O perceptron errou algum dado de treinamento')

print('Pesos após treinamento: ',ptn.intercept_, ptn.coef_)
print('Número de iterações: ',ptn.n_iter_)

T_teste = [1,1,1,1,1,1,1,0,1,0]
H_teste = [1,1,0,0,1,1,1,1,0,1] 

print('Testando o perceptron com o novo formato de T: ',ptn.predict([T_teste]))
print('Testando o perceptron com o novo formato de H: ',ptn.predict([H_teste]))