from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

'''
Works in:
English, French, German,
Dutch, Italian, Spanish
'''
'''
Other sentiment models:
Provides sentiment 1-5: nlptown/bert-base-multilingual-uncased-sentiment
Provides many emotions: SamLowe/roberta-base-go_emotions
Provides Pos/Neu/Neg: cardiffnlp/twitter-roberta-base-sentiment-latest
'''

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def get_sentiment_score(text):
    tokens = tokenizer.encode(text, return_tensors='pt')
    result = model(tokens)
    # Get the sentiment (0,1,2,3,4) and +1 to get 1-5 score
    sentiment = int(torch.argmax(result.logits))+1
    return sentiment