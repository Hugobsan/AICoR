import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn import functional as F
import torch

# Carregando os dados do arquivo CSV
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Dividindo os dados em treinamento e teste
train_data, test_data, train_labels, test_labels = train_test_split(
    df['Texto'], df['Sentimento'], test_size=0.2, random_state=42
)

# Tokenizando os textos
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_data), truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(list(test_data), truncation=True, padding=True, max_length=128, return_tensors='pt')

# Convertendo rótulos para tensores
train_labels = torch.tensor(train_labels.values)
test_labels = torch.tensor(test_labels.values)

# Criando conjuntos de dados PyTorch
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

# Definindo o modelo BERT para classificação de sequências
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Parâmetros de treinamento
optimizer = AdamW(model.parameters(), lr=5e-5)
batch_size = 8
epochs = 3

# Dividindo o conjunto de treinamento para treinamento e validação
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Criando DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Treinamento do modelo
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Avaliação do modelo
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = F.softmax(logits, dim=1).argmax(dim=1).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# Avaliação e métricas
accuracy = accuracy_score(all_labels, all_preds)
classification_rep = classification_report(all_labels, all_preds)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')
