from utils import detect_emotion, extract_keywords, get_semantic_similarity

def recommend_verses(query, verses_df):
    # Extract columns for processing
    verses = verses_df[[
        'Title', 'Chapter', 'Verse', 'Enlgish Translation', 'Explanation',
        'Emotional Themes', 'Keywords Tags'
    ]]

    # Detect emotion
    emotion = detect_emotion(query)

    # Extract keywords
    keywords = extract_keywords(query)

    # Get semantic similarity scores
    semantic_scores = get_semantic_similarity(query, verses['Enlgish Translation'].tolist())

    # Combine results with weights
    recommendations = []
    for idx, verse in verses.iterrows():
        emotion_score = 0.3 if emotion in verse['Emotional Themes'] else 0
        keyword_overlap = len(set(keywords) & set(verse['Keywords Tags'].split(',')))
        keyword_score = 0.3 * (keyword_overlap / max(len(keywords), 1))
        semantic_score = 0.4 * semantic_scores[idx]

        total_score = emotion_score + keyword_score + semantic_score
        recommendations.append((verse.to_dict(), total_score))

    # Sort by scores
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:3]
    return [rec[0] for rec in recommendations]
