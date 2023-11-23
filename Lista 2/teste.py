import random

class Perceptron:
    #Método construtor do perceptron
    def __init__(self, inputSize, LrnRate, bias, defaultWeights):
        self.weights = []
        self.bias = bias
        self.trainingIterations = 0
        self.learningRate = LrnRate

        #Caso não seja passado um valor default para os pesos, eles serão gerados aleatoriamente
        if(defaultWeights):
            self.weights = defaultWeights
        else:
            for i in range(inputSize):
                self.weights.append(round(random.random(), 2))

    #Método que retorna a saída baseado nas entradas e pesos atuais    
    def predict(self, inputs):
        sum = 0

        for i in range(inputs.__len__()):
            sum += inputs[i] * self.weights[i]
        
        return 0 if sum < self.bias else 1
    
    #Método que treina o perceptron com os dados de entrada e saída esperada
    def train(self, inputs, target):
        guess = self.predict(inputs)
        error = target - guess

        for i in range(self.weights.__len__()):
            #adicionando o novo peso truncando com 4 casas decimais
            self.weights[i] = round(self.weights[i] + self.learningRate * error * inputs[i], 4)

print("---- Algoritmo Perceptron ----")
print("\n")

#Injetando configurações passadas na atividade
perceptron = Perceptron(inputSize=3, LrnRate=0.1, defaultWeights=[-0.5, 0, 0], bias=0)

#Definindo entradas de treinamento
training_entries = [
  { "inputs": [1, 0, 0], "label": 0 },
  { "inputs": [1, 0, 1], "label": 1 },
  { "inputs": [1, 1, 0], "label": 1 },
  { "inputs": [1, 1, 1], "label": 1 }
]

#Exibindo algumas informações do problema
print("--------------------------------")
print("Número de entradas: ", perceptron.weights.__len__())
print("Pesos iniciais: ", perceptron.weights)
print("\n")
print("Entradas de treinamento: ")
for entry in training_entries:
    print(entry.get("inputs"), " -> ", entry.get("label"))
print("")
print("--------------------------------")

while True:
    ##Rodando um ciclo de treinamento do modelo
    for entry in training_entries:
        perceptron.train(entry.get("inputs"), entry.get("label"))
    
    perceptron.trainingIterations += 1
    print("Iteração:", perceptron.trainingIterations, ", Pesos:", perceptron.weights)
    finish = 0
    
    ##Testando o modelo. Caso o modelo acerte todas as entradas de treinamento, o loop é finalizado
    for entry in training_entries:
        if(entry.get("label") == perceptron.predict(entry.get("inputs"))): 
            finish += 1
    if(finish == training_entries.__len__()):
        break
    
print("--------------------------------")
print("Treinamento Finalizado!")
print("Número de Iterações: ", perceptron.trainingIterations)
print("Pesos Finais: ", perceptron.weights)
print("--------------------------------")

print("\n")

input("Pressione ENTER para sair...")