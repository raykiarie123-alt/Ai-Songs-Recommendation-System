from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import difflib


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)
songs_data=pd.read_csv("spotify_millsongdata.csv")
#take the necessary columns
selected_features = ["artist", "song", "text"]

for feature in selected_features:
  songs_data[feature] = songs_data[feature].fillna('')

combined_features = songs_data["artist"] + " " + songs_data["song"] + " " + songs_data["text"]
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

songs_index = 10  # example

similarity = cosine_similarity(
    feature_vectors[songs_index],
    feature_vectors
)
list_of_all_titles = songs_data['song'].to_list()

#form to maintain user input
class SongRequest (BaseModel):
  song:str

@app.get("/")
def root():
  return{"message": "Songs Recommendation API"}

@app.post("/")
def recommend (request: SongRequest):
  song_name = request.song
  find_close_match = difflib.get_close_matches(song_name, list_of_all_titles)
  close_match = find_close_match
  index_of_the_song = songs_data[songs_data.title == close_match]['index'].values[0]
  similarity_score = list(enumerate(similarity[index_of_the_song]))
  sorted_similar_songs = sorted(similarity_score, key=lambda x:x [1], reverse=True)

def get_song_recommendations(sorted_similar_songs, songs_data, close_match):
    recommendations = []

    for i, song in enumerate(sorted_similar_songs[1:20]):
        index = song[0]
        song_name = songs_data.loc[index, 'song']
        recommendations.append(song_name)

    return {
        "matched_songs": close_match,
        "recommendations": recommendations
    }
