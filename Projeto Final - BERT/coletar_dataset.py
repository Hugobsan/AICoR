import tweepy
import csv

# Configure suas chaves de API
consumer_key = 'sua_consumer_key'
consumer_secret = 'sua_consumer_secret'
access_token = 'seu_access_token'
access_token_secret = 'seu_access_token_secret'

# Autentique-se na API do Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Realize uma busca por tweets com a palavra-chave "python"
query = 'python'
num_tweets = 5000

# Lista para armazenar os tweets
tweets_list = []

# Coleta de tweets
for tweet in tweepy.Cursor(api.search, q=query, lang='en').items(num_tweets):
    # Obtenha a polaridade do Sentiment140 (0 ou 4)
    polarity = 0 if tweet.favorite_count == 0 else 4
    
    # Converta a polaridade para "Negativo" ou "Positivo"
    sentiment = 'Negativo' if polarity == 0 else 'Positivo'
    
    tweets_list.append([tweet.id, tweet.created_at, tweet.text, sentiment])

# Salvar os tweets em um arquivo CSV
csv_file_path = 'tweets_python_sentiment140.csv'
header = ['ID', 'Data de Criação', 'Texto', 'Sentimento']

with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(tweets_list)

print(f'{num_tweets} tweets foram coletados e salvos em {csv_file_path}.')
