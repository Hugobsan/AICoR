##Modificação de parâmetros
BIN_PATH = 'trained.bin'

#Nome do modelo pré treinado para tokenização das entradas
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

#Tamanho máximo de tokens (Tamanho máximo das reviews após tokenização)
MAX_LEN = 160

#Dispoitivo de processamento de preferência (CPU ou GPU) (Deixe em None e o código escolherá o melhor dispositivo)
FAVORITE_DEVICE = None

#Valor do Dropout
DROPOUT = 0.3

#Nomes das classes de sentimento
CLASSES = ['negativo', 'neutro', 'positivo']

from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import warnings
# Suprimindo warnings chatos
warnings.filterwarnings("ignore")

print("-"*50)
print("Implementação de um modelo de classificação de sentimentos utilizando BERT")
print("-"*50)

#importando um modelo pré treinado para tokenizar as entradas
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, truncation=True)

if FAVORITE_DEVICE is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    if FAVORITE_DEVICE == "cpu":
        device = torch.device("cpu")
    elif FAVORITE_DEVICE == "gpu":
        device = torch.device("cuda:0")

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
    self.drop = nn.Dropout(p=DROPOUT)
    #The last_hidden_state is a sequence of hidden states of the last layer of the model
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

class_names = CLASSES
model = SentimentClassifier(len(class_names))
print("Carregando modelo...")
# Caso exista um arquivo de rede já treinada
if torch.cuda.is_available():
  model.load_state_dict(torch.load(BIN_PATH))
else:
  model.load_state_dict(torch.load(BIN_PATH, map_location=torch.device('cpu')))
model = model.to(device) #Coloca o modelo na GPU

def predict_sentiment(text):
  encoded_review = tokenizer.encode_plus(
    text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
    truncation=True
  )

  input_ids = encoded_review['input_ids'].to(device)
  attention_mask = encoded_review['attention_mask'].to(device)

  output = model(input_ids, attention_mask)
  _, prediction = torch.max(output, dim=1)

  probs = F.softmax(output, dim=1) #Softmax gera probabilidades para cada classe

  print("\n========================")
  print(f'Texto analisado: {text}')
  print(f'Sentimento: {class_names[prediction]}')
  print('Probabilidades:')
  print(pd.DataFrame(probs.tolist()[0], class_names)[0])
  print("========================\n")

while True:
  print("Digite uma frase positiva, negativa ou neutra (exit para sair): ")
  test = input()
  if test == "exit":
      break
  predict_sentiment(test)
