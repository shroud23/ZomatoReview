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

# Load your dataset from the provided path
data = pd.read_csv('C:\\Users\\Utkarsh Yadav\\OneDrive\\Desktop\\project\\project 1\\reviews.csv')

# Map 'overall rating' to sentiment labels (for example, you can consider ratings above a threshold as 'positive')
threshold = 3.0
data['sentiment'] = data['overall rating'].apply(lambda x: 'positive' if x >= threshold else 'negative')

# Apply preprocessing to the 'review' column
data['review'] = data['review'].apply(preprocess_text)

# Save the preprocessed data
data.to_csv('preprocessed_data.csv', index=False)
