# MOVIE RECOMMENDATION SYSTEM

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv(
    r"C:\Users\fatai\Desktop\ML AND AI Learning\Top_10000_Movies.csv",
    engine='python',
    on_bad_lines='skip'
)   #Update with your path

# Display first few rows
print(data.head())

# Check info
print("\n Dataset Info:")
print(data.info())

# Keep useful columns and drop missing values
movies = data[['original_title', 'genre', 'overview', 'tagline', 'vote_average', 'popularity']].dropna()

# Combine text-based features
movies['combined_features'] = (
    movies['genre'] + ' ' +
    movies['overview'] + ' ' +
    movies['tagline']
)

# Vectorize text features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined_features'])

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Function to recommend movies
def recommend(movie_title, num_recommendations=5):
    if movie_title not in movies['original_title'].values:
        print("Movie not found in dataset.")
        return

    # Get index of the given movie
    idx = movies[movies['original_title'] == movie_title].index[0]

    # Compute similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations * 3]

    # Get top candidates
    movie_indices = [i[0] for i in sim_scores]
    recommended = movies.iloc[movie_indices][['original_title', 'vote_average', 'popularity']].copy()

    # Combine similarity and quality score
    recommended['final_score'] = (
        recommended['vote_average'] * 0.6 + 
        recommended['popularity'] * 0.4
    )

    # Sort by final combined score
    recommended = recommended.sort_values(by='final_score', ascending=False).head(num_recommendations)

    print(f"\nBecause you watched '{movie_title}', you might also like:\n")
    print(recommended[['original_title', 'vote_average', 'popularity']].to_string(index=False))

# Example
recommend("Avatar")  # Change this to any movie title in your dataset
