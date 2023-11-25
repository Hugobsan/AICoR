import bert

BIN_PATH = '/content/drive/MyDrive/Documentos/Documentos Faculdade/Sistemas de Informação/6º Período/IA/Projeto Final/24-11-2023_03-38-40 - df=21600 - accuracy=0.75.bin'

bert = bert.BertSentimentClassifier(binary_path=BIN_PATH)

CSV_FILE = '/content/drive/MyDrive/Documentos/Documentos Faculdade/Sistemas de Informação/6º Período/IA/Projeto Final/reviews.csv'

df = bert.generate_subset(CSV_FILE)

bert.generate_token_lens()
