import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Data preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load your dataset
data = pd.read_csv('C:\\Users\\Utkarsh Yadav\\OneDrive\\Desktop\\project\\project 1\\reviews.csv')

# Convert 'Overall_Rating' column to numeric
data['Overall_Rating'] = pd.to_numeric(data['Overall_Rating'], errors='coerce')  # 'coerce' handles non-numeric values gracefully

# Use 'Overall_Rating' column to generate sentiment labels
def assign_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating < 2:
        return 'negative'
    else:
        return 'neutral'

data['sentiment'] = data['Overall_Rating'].apply(assign_sentiment)

# Preprocess the text column (assuming 'Review' is the text column)
data['Review'] = data['Review'].apply(preprocess_text)

# Save the preprocessed data
data.to_csv('preprocessed_data.csv', index=False)
