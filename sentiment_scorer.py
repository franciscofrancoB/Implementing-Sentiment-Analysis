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

sentiment_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
BERT_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def get_sentiment_score(text):
    tokens = sentiment_tokenizer .encode(text, return_tensors='pt')
    model_output = BERT_model(tokens)
    # Get the sentiment (0,1,2,3,4) and +1 to get 1-5 score
    sentiment_score = int(torch.argmax(model_output.logits))+1
    return sentiment_score