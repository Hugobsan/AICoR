import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import warnings
# Suprimindo warnings chatos
warnings.filterwarnings("ignore")

print("-"*50)
print("Implementação de um modelo de classificação de sentimentos utilizando BERT")
print("-"*50)
print("Alunos: Hugo Barbosa Santos e Bernardo Mendes Pereira")
print("-"*50)
print("\n\n")


BIN_PATH = 'best_model_state.bin'
#Nome do modelo pré treinado
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

print("Modelo pré treinado utilizado: ", PRE_TRAINED_MODEL_NAME)
print("\n\n")

#importando um modelo pré treinado para tokenizar as entradas
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, truncation=True)

# Carregue os dados do CSV
def generate_subset(file_location, desired_count = None):
    df = pd.read_csv(file_location)

    print(f"Tamanho total do dataset: {len(df)}")

    #caso desired_count seja None receber a quantidade de valores no csv / 3
    if desired_count is None:
      desired_count = len(df) // 3

    # Inicialize um dataframe vazio
    df_subset = pd.DataFrame()

    # Para cada valor de "sentiment", pegue os primeiros 'desired_count' dados
    for sentiment_value in range(3):
        df_sentiment = df[df["sentiment"] == sentiment_value].head(desired_count)
        df_subset = pd.concat([df_subset, df_sentiment], ignore_index=True)

    # Embaralhe os dados (opcional)
    df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Quantidade de dados por sentimento: ")
    print("0 - Negativo: ", df_subset[df_subset["sentiment"] == 0].shape[0])
    print("1 - Neutro: ", df_subset[df_subset["sentiment"] == 1].shape[0])
    print("2 - Positivo: ", df_subset[df_subset["sentiment"] == 2].shape[0])
    print("\n\n")

    return df_subset

CSV_FILE = 'reviews.csv'
df = generate_subset(CSV_FILE)

# Plotar a distribuição de tokens
token_lens = []

print("Plotando a distribuição de tokens...")
for txt in df.content:
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 200]);
plt.xlabel('Token count');

MAX_LEN = 160
print("Tamanho máximo de tokens: ", MAX_LEN)

# Criando um dataset customizado
# Uma seed é um número aleatório que é usado para inicializar um gerador de números aleatórios.
# A seed é definida para garantir a reprodutibilidade dos resultados (mesma saída para cada execução)
print("Criando um dataset customizado...")
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Classe para criar um dataset customizado
class GPReviewDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      #padding='longest',
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

# Dividindo o dataset em treino, validação e teste
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

print("Dataset dividido em 70% treino, 15% validação e 15% teste")

df_train.shape, df_val.shape, df_test.shape

# Criando dataloaders
print("Criando dataloaders...")
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.content.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len,
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

print("Parâmetros:")

data = next(iter(train_data_loader))
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

# Define o dispositivo de execução como "cuda:0" se houver uma GPU disponível, caso contrário, usa a CPU. 
# Isso permite que o código seja executado de forma otimizada em hardware acelerado por GPU, se disponível.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
    self.drop = nn.Dropout(p=0.3)
    #The last_hidden_state is a sequence of hidden states of the last layer of the model
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

class_names = ['negativo', 'neutro', 'positivo']
model = SentimentClassifier(len(class_names))
# Caso exista um arquivo de rede já treinada
try:
  model.load_state_dict(torch.load(BIN_PATH))
except:
  None  
model = model.to(device) #Coloca o modelo na GPU

##Treinamento

#Definindo ciclos de treinamento
EPOCHS = 3

#Configurando o otimizador e o scheduler
optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

#Definindo a função de perda
loss_fn = nn.CrossEntropyLoss().to(device)

#Função para treinar um ciclo
def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

#Função para avaliar um ciclo
def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

###Realizando treinamento
print("Realizando o treinamento...")

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(df_val)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    #Salvando os pesos do modelo caso a acurácia seja maior que a anterior
    from datetime import datetime
    filename = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + " - df=" + str(len(df)) +" - accuracy=" + val_acc + ".bin"
    torch.save(model.state_dict(), filename)
    best_accuracy = val_acc
## Plotando os resultados
if torch.cuda.is_available():
    history['train_acc'] = [h.cpu() for h in history['train_acc']]
    history['val_acc'] = [h.cpu() for h in history['val_acc']]

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);

##Avaliação das métricas
print("Avaliação das métricas...")
def get_predictions(model, data_loader):
  model = model.eval()

  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)

print("MATRIZ DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_pred, target_names=class_names))

def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment');

print("MATRIZ CONFUSÃO:")
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

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
##Testando a rede
print("Testando a rede...")
while True:
  test = input("Digite uma frase positiva, negativa ou neutra (exit para sair ou default para testar uma frase do dataset): ")
  if test == "exit":
      break
  elif test == "default":
      rand = np.random.randint(0, len(df_test))
      test = df_test.iloc[rand].content
  predict_sentiment(test)