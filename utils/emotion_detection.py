from transformers import pipeline

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
def detect_emotion(query):
    result = emotion_model(query)
    return result[0]['label']
