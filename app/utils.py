from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# Load models
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
semantic_model = SentenceTransformer('all-mpnet-base-v2')

def detect_emotion(query):
    result = emotion_model(query)
    return result[0]['label']

def extract_keywords(query):
    # Simplified TF-IDF approach
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([query])
    keywords = vectorizer.get_feature_names_out()
    return keywords[:3]

def get_semantic_similarity(query, corpus):
    query_embedding = semantic_model.encode(query)
    corpus_embeddings = semantic_model.encode(corpus)
    scores = util.cos_sim(query_embedding, corpus_embeddings).squeeze(0).tolist()
    return scores
