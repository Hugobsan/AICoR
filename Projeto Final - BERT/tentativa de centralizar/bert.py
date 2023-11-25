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

class BertSentimentClassifier():
    def __init__(self, 
                 binary_path='best_model_state.bin', 
                 pretrained_model_name='neuralmind/bert-base-portuguese-cased',
                 random_seed = 42
                ):
        self.binary_path = binary_path
        self.classes = ['negativo', 'neutro', 'positivo']
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.file_location = None
        self.df = None
        self.token_lens = None
        self.max_len = None
        self.random_seed = random_seed
        self.num_workers = 4

    def generate_subset(self, file_location, desired_count = None):
        self.file_location = file_location
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

        print(f"Tamanho do dataset subset: {len(df_subset)}")
        print("Quantidade de dados por sentimento: ")
        print("0 - Negativo: ", df_subset[df_subset["sentiment"] == 0].shape[0])
        print("1 - Neutro: ", df_subset[df_subset["sentiment"] == 1].shape[0])
        print("2 - Positivo: ", df_subset[df_subset["sentiment"] == 2].shape[0])
        
        self.df = df_subset
        return df_subset

    def generate_token_lens(self, max_len = 512):
        # Plotar a distribuição de tokens
        token_lens = []

        for txt in self.df.content:
            tokens = self.tokenizer.encode(txt, max_length=max_len, truncation=True)
            token_lens.append(len(tokens))

        sns.distplot(token_lens)
        plt.xlim([0, 200]);
        plt.xlabel('Token count');

        self.max_len = 160

        # Criando um dataset customizado
        # Uma seed é um número aleatório que é usado para inicializar um gerador de números aleatórios.
        # A seed é definida para garantir a reprodutibilidade dos resultados (mesma saída para cada execução)
        RANDOM_SEED = self.random_seed

        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    def split_set(self, test_size = 0.1, validation_size=0.5):
       # Dividindo o dataset em treino, validação e teste
        test_size2 = 1-validation_size

        df_train, df_test = train_test_split(self.df, test_size=test_size, random_state=self.random_seed)
        df_val, df_test = train_test_split(df_test, test_size=test_size2, random_state=self.random_seed)

        print(f"Dataset dividido em {(1-test_size)*100}% treino, {(test_size*(1-test_size2))*100}% validação e {test_size*test_size2*100}% teste")

        df_train.shape, df_val.shape, df_test.shape

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = GPReviewDataset(
            reviews=df.content.to_numpy(),
            targets=df.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.num_workers
        )


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


        

        
    