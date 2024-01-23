import pandas as pd
from sentiment_scorer import get_sentiment_score

# Sample sentences
data = {
    'sentences': [
        'The sunset over the ocean was breathtaking.',
        'I am deeply disappointed with the poor service at the restaurant.',
        'Winning the award was the happiest moment of my life.',
        'The constant noise from the construction site is unbearable.',
        'I am so excited about my upcoming vacation to Hawaii.',
        'The book was so dull that I couldn\'t even finish it.',
        'I feel so energized after that amazing workout session.',
        'The loss of my pet has left me feeling heartbroken.',
        'The cake I baked turned out to be incredibly delicious.',
        'Dealing with the insurance company has been a nightmare.'
    ]
}

df = pd.DataFrame(data)

# Sentiment scoring function
df['sentiment_score'] = df['sentences'].apply(get_sentiment_score)

print(df)
