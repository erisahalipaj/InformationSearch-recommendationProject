import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self):
        self.movies = None
        self.genome_scores = None
        self.ratings = None
        self.movies_with_content = None
        self.similarity_matrix = None

    def load_data(self):
        self.genome_scores = pd.read_csv("D:/AAU_UDINE/Semester1/RecommendationSystems/ml-25m/ml-25m/genome-scores.csv")
        self.movies = pd.read_csv('D:/AAU_UDINE/Semester1/RecommendationSystems/ml-25m/ml-25m/movies.csv')
        self.ratings = pd.read_csv('D:/AAU_UDINE/Semester1/RecommendationSystems/ml-25m/ml-25m/ratings.csv')

        # Calculate mean rating and the number of ratings for each movie
        movie_ratings = self.ratings.groupby('movieId').agg({'rating': ['mean', 'count']}).reset_index()
        movie_ratings.columns = ['movieId', 'mean_rating', 'rating_count']

        # Merge movie details with ratings
        self.movies_with_ratings = pd.merge(self.movies, movie_ratings, on='movieId', how='left')

        # Pivot genome_scores to create a matrix with movieId as rows and tagId as columns
        genome_matrix = self.genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)

        # Merge genome_matrix with movie details
        self.movies_with_content = pd.merge(self.movies_with_ratings, genome_matrix, on='movieId', how='left')

    def calculate_item_similarity(self):
        # Drop non-feature columns for similarity calculation
        content_features = self.movies_with_content.drop(
            columns=['movieId', 'title', 'genres', 'mean_rating', 'rating_count'])

        # Ensure there are no NaN values in the features
        content_features = content_features.fillna(0)

        # Calculate cosine similarity matrix
        self.similarity_matrix = cosine_similarity(content_features)

    def recommend_similar_movies(self, input_movie_name, top_n=10):
        # Find the input movie's id
        input_movie = self.movies_with_content[
            self.movies_with_content['title'].str.contains(input_movie_name, case=False, na=False)]
        if input_movie.empty:
            raise ValueError("Movie not found in the dataset.")

        input_movie_id = input_movie.iloc[0]['movieId']

        # Get index of the input movie in the similarity matrix
        movie_index = self.movies_with_content.index[self.movies_with_content['movieId'] == input_movie_id][0]

        # Get similarity scores for the input movie
        similarity_scores = list(enumerate(self.similarity_matrix[movie_index]))

        # Sort movies based on similarity scores
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get top N similar movies
        similar_movies_indices = [i[0] for i in similarity_scores[1:top_n + 1]]  # Exclude the input movie itself

        # Create a DataFrame with similar movie details
        recommendations = self.movies_with_content.iloc[similar_movies_indices]

        # Adjust similarity score by popularity (mean_rating * rating_count)
        recommendations['popularity_score'] = recommendations['mean_rating'] * recommendations['rating_count']
        recommendations['adjusted_similarity'] = recommendations.apply(
            lambda x: x['popularity_score'] * similarity_scores[similar_movies_indices.index(x.name)][1], axis=1)

        # Sort by the adjusted similarity score and get the top_n recommendations
        recommendations = recommendations.sort_values(by='adjusted_similarity', ascending=False).head(top_n)

        return recommendations[['title', 'mean_rating', 'rating_count', 'adjusted_similarity']]


def main():
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.calculate_item_similarity()
    movie_title = input("Enter Movie Title: ")
    try:
        recommendations = recommender.recommend_similar_movies(movie_title)
        print("Top 10 Similar Movie Recommendations:\n", recommendations)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
