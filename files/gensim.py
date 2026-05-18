from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re



reviews = [
    "The delivery was very fast. The product was well packed.",
    "The laptop works perfectly. The battery lasts a long time.",
    "Customer service was excellent. Very helpful staff.",
    "The item arrived scratched. The box was damaged.",
    "Very bad service. Support never replied.",
    "The headphones have good sound quality. They are slightly uncomfortable.",
    "The price matches the quality. I am satisfied with the purchase.",
    "The application crashes frequently after the update.",
    "Everything works normally. Nothing special.",
    "The courier was two days late.",
    "The smartphone design looks modern and stylish.",
    "The build quality is average. I expected something better.",
    "After one month of use. The phone stopped charging.",
    "The mouse is comfortable for work and gaming.",
    "An ordinary product for its price.",
    "Support helped solve the issue quickly.",
    "The screen has dead pixels.",
    "The system runs stable. However, the interface is inconvenient.",
    "I am very happy with the purchase. Highly recommended.",
    "The product does not match the description on the website."
]

def preprocess(text):
    text = re.sub(
        r"[^\w\s]", "", text.lower()
    )
    return text.split()


tokenized_corpus = [preprocess(sentence) for sentence in corpus]