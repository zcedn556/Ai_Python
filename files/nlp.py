import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

nltk.download("punkt")  # Для токенізації
nltk.download("punkt_tab")  # Для токенізації речень
nltk.download("stopwords")  # Стоп-слова
nltk.download("wordnet")  # Для лемматизації
nltk.download("omw-1.4")  # WordNet мовні

df = pd.read_csv('./csv/reviews.csv')

for i in range(20):
    sentences = sent_tokenize(df.iloc[i,1])
    print(sentences)

filtered_words = []

stop_words = set(stopwords.words("english"))
for reviews in df["review"]:
    words = word_tokenize(reviews)
    
    filtered_words.extend([
        word for word in words
        if word.lower() not in stop_words and word.isalpha()
    ])

    print(filtered_words)

stemmer = PorterStemmer()
stemmed_words = []
print("stemming \n\n")
for word in filtered_words:
    stemmed_words = [stemmer.stem(word)]

    print(stemmed_words)

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in filtered_words:
    lemmatized_words.append(lemmatizer.lemmatize(word))

    print(lemmatized_words)


for blob in df["review"]:
    review = TextBlob(blob)
    sentiment = review.sentiment.polarity
    if sentiment > 0.2:
        print("Good")
    elif -0.2 <= sentiment <= 0.2:
        print("Middle")
    else:
        print("bad")