from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample dataset: Movies and their genres
movies = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ["Inception", "Titanic", "Interstellar", "Avatar", "The Matrix"],
    'genres': ["Sci-Fi Thriller", "Romance Drama", "Sci-Fi Adventure", "Sci-Fi Action", "Sci-Fi Action"]
})

# User rating data
ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'movie_id': [1, 2, 2, 3, 3, 4, 4, 5, 5, 1],
    'rating': [5, 4, 5, 4, 3, 5, 4, 5, 4, 4]
})

### 📌 1️⃣ Content-Based Filtering (Using Genres)
def content_based_recommendations(movie_title, top_n=3):
    """Recommend movies based on genre similarity."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    if movie_title not in movies["title"].values:
        return ["Movie not found in the dataset."]
    
    movie_index = movies.index[movies["title"] == movie_title].tolist()[0]
    similar_movies = list(enumerate(similarity[movie_index]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    return [movies.iloc[i[0]]["title"] for i in similar_movies]

### 📌 2️⃣ Collaborative Filtering (User-Based & Item-Based)
ratings_matrix = ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)

# Compute user-user similarity
user_similarity = ratings_matrix.corr(method="pearson")

# Compute item-item similarity
item_similarity = ratings_matrix.T.corr(method="pearson")

def user_based_recommendations(user_id, top_n=3):
    """Recommend movies based on similar users' preferences."""
    if user_id not in ratings_matrix.index:
        return ["User ID not found in the dataset."]
    
    similar_users = user_similarity[user_id].sort_values(ascending=False)[1:top_n+1]
    
    recommendations = []
    for sim_user in similar_users.index:
        top_movies = ratings_matrix.loc[sim_user][ratings_matrix.loc[sim_user] > 4].index.tolist()
        recommendations.extend(top_movies)
    
    recommendations = list(set(recommendations))[:top_n]
    return movies[movies["movie_id"].isin(recommendations)]["title"].tolist()

def item_based_recommendations(movie_title, top_n=3):
    """Recommend similar movies based on item-item similarity."""
    if movie_title not in movies["title"].values:
        return ["Movie not found in the dataset."]
    
    movie_id = movies[movies["title"] == movie_title]["movie_id"].values[0]
    similar_items = item_similarity[movie_id].sort_values(ascending=False)[1:top_n+1]
    
    return movies[movies["movie_id"].isin(similar_items.index)]["title"].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        rec_type = request.form["rec_type"]
        
        if rec_type == "content":
            movie_name = request.form["movie_name"]
            recommendations = content_based_recommendations(movie_name)
        
        elif rec_type == "user":
            user_id = int(request.form["user_id"])
            recommendations = user_based_recommendations(user_id)
        
        elif rec_type == "item":
            movie_name = request.form["movie_name"]
            recommendations = item_based_recommendations(movie_name)

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
