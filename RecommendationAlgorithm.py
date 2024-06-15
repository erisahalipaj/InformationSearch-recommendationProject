import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class MovieRecommender:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.item_similarity = None

    def load_data(self):
        movies_path = "D:/AAU_UDINE/Semester1/RecommendationSystems/ml-25m/ml-25m/movies.csv"
        ratings_path = "D:/AAU_UDINE/Semester1/RecommendationSystems/ml-25m/ml-25m/ratings.csv"

        # Load movies data
        self.movies = pd.read_csv(movies_path, encoding='latin1')
        print("First movie in the dataset:", self.movies.iloc[0])

        # Load ratings data
        self.ratings = pd.read_csv(ratings_path, encoding='latin1')
        print("Ratings data loaded. Sample:")
        print(self.ratings.head())

        # Downsample the ratings data for testing purposes
        self.ratings = self.ratings.sample(frac=0.05, random_state=42)  # Further downsampling to 5%
        print("Downsampled ratings data loaded. Sample:")
        print(self.ratings.head())

        # Filter movies with at least 50 ratings and users who have rated at least 50 movies
        movie_counts = self.ratings['movieId'].value_counts()
        user_counts = self.ratings['userId'].value_counts()
        self.ratings = self.ratings[self.ratings['movieId'].isin(movie_counts[movie_counts >= 50].index)]
        self.ratings = self.ratings[self.ratings['userId'].isin(user_counts[user_counts >= 50].index)]

        print("Filtered ratings data loaded. Sample:")
        print(self.ratings.head())

    def calculate_item_similarity(self):
        # Create a pivot table
        movie_ratings = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        # Create a sparse matrix
        movie_ratings_matrix = csr_matrix(movie_ratings.values)

        # Compute cosine similarity between items
        self.item_similarity = cosine_similarity(movie_ratings_matrix.T)

    def recommend_similar_movies(self, movie_title):
        # Get the movie ID for the given movie title
        if movie_title not in self.movies['title'].values:
            print(f"Movie '{movie_title}' not found in the dataset.")
            return []

        movie_id = self.movies[self.movies['title'] == movie_title]['movieId'].values[0]

        # Find the index of the movie
        movie_idx = self.movies[self.movies['movieId'] == movie_id].index[0]

        # Get similarity scores for the movie
        similarity_scores = self.item_similarity[movie_idx]

        # Create a list of tuples (movie_id, similarity_score)
        similar_movies = [(self.movies.iloc[idx]['title'], score) for idx, score in enumerate(similarity_scores) if idx != movie_idx]

        # Sort the movies based on similarity score
        similar_movies.sort(key=lambda x: x[1], reverse=True)

        # Get the top 10 similar movies
        top_similar_movies = similar_movies[:10]

        # Return the titles of the top 10 similar movies
        return [title for title, score in top_similar_movies]

def main():
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.calculate_item_similarity()
    movie_title = input("Enter Movie Title: ")
    try:
        recommendations = recommender.recommend_similar_movies(movie_title)
        print("Top 10 Similar Movie Recommendations:", recommendations)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
