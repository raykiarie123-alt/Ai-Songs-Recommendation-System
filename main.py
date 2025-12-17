from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import difflib

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title="Song Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load and prepare data (runs once at startup)
# --------------------------------------------------
songs_data = pd.read_csv("spotify_millsongdata.csv")

# Columns used for recommendations
selected_features = ["artist", "song", "text"]

# Handle missing values
for feature in selected_features:
    songs_data[feature] = songs_data[feature].fillna("")

# Combine text features
combined_features = (
    songs_data["artist"] + " " +
    songs_data["song"] + " " +
    songs_data["text"]
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
feature_vectors = vectorizer.fit_transform(combined_features)

# List of song titles (for matching user input)
list_of_all_titles = songs_data["song"].tolist()

# --------------------------------------------------
# Request model
# --------------------------------------------------
class SongRequest(BaseModel):
    song: str

# --------------------------------------------------
# Root endpoint
# --------------------------------------------------
@app.get("/")
def root():
    return {"message": "Songs Recommendation API is running 🚀"}

# --------------------------------------------------
# Recommendation endpoint
# --------------------------------------------------
@app.post("/")
def recommend(request: SongRequest):
    user_song = request.song

    # Find closest matching song name
    close_matches = difflib.get_close_matches(
        user_song,
        list_of_all_titles,
        n=1,
        cutoff=0.6
    )

    if not close_matches:
        raise HTTPException(
            status_code=404,
            detail="Song not found in database"
        )

    matched_song = close_matches[0]

    # Get index of matched song
    song_index = songs_data[
        songs_data["song"] == matched_song
    ].index[0]

    # Compute similarity for this song
    similarity_scores = cosine_similarity(
        feature_vectors[song_index],
        feature_vectors
    )[0]

    # Enumerate & sort similarity scores
    similarity_score = list(enumerate(similarity_scores))
    sorted_similar_songs = sorted(
        similarity_score,
        key=lambda x: x[1],
        reverse=True
    )

    # Collect recommendations
    recommendations = []
    for index, score in sorted_similar_songs[1:20]:
        recommendations.append(songs_data.loc[index, "song"])

    return {
        "matched_song": matched_song,
        "recommendations": recommendations
    }
