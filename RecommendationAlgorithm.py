import pandas as pd

class MovieRecommender:
    def __init__(self):
        self.users = None
        self.movies = None
        self.ratings = None
        self.merged_data = None
        self.LIKE_THRESHOLD = 3  # Define a constant for the "like" rating threshold

    def load_data(self):
        # Define the path to the dataset files
        users_path = r'C:\Users\USER\OneDrive\Documents\Semester 2\Information Search & Recommendation Systems\HW\Assignment4\users.dat'
        movies_path = r'C:\Users\USER\OneDrive\Documents\Semester 2\Information Search & Recommendation Systems\HW\Assignment4\movies.dat'
        ratings_path = r'C:\Users\USER\OneDrive\Documents\Semester 2\Information Search & Recommendation Systems\HW\Assignment4\ratings.dat'

        # Load users data
        users_columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
        self.users = pd.read_csv(users_path, sep='::', engine='python', names=users_columns, encoding='latin1')

        # Load movies data
        movies_columns = ['movie_id', 'title', 'genres']
        self.movies = pd.read_csv(movies_path, sep='::', engine='python', names=movies_columns, encoding='latin1')

        # Load ratings data
        ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=ratings_columns, encoding='latin1')

        # Merge the datasets
        self.merged_data = pd.merge(pd.merge(self.ratings, self.users, on='user_id'), self.movies, on='movie_id')

    def get_user_ratings(self, user_id):

        user_ratings = self.merged_data[self.merged_data['user_id'] == int(user_id)]

        if user_ratings.empty:
            print('No ratings for user {}'.format(user_id))
        else:
            print(user_ratings[['title', 'rating', 'genres']])
        return user_ratings
    def create_user_profile(self, user_id):
        user_ratings = self.get_user_ratings(user_id)
        liked_movies = user_ratings[user_ratings['rating'] > self.LIKE_THRESHOLD]

        genre_counts = {}
        for genres in liked_movies['genres']:
            for genre in genres.split('|'):
                if genre in genre_counts:
                    genre_counts[genre] += 1
                else:
                     genre_counts[genre] = 1
        print("User Profile based on liked genres:", genre_counts)
        return genre_counts

    def calculate_genre_overlap(self,movies_genres,user_profile):
        movies_genres_set = set(movies_genres.split('|'))
        user_genre_set = set(user_profile.keys())
        overlap = movies_genres_set.intersection(user_genre_set)
        return len(overlap)

    def recommend_movies(self, user_id):
        user_profile = self.create_user_profile(user_id)
        if not user_profile:
            return []

        # Calculate popularity of each movie based on how often it's rated
        movie_popularity = self.ratings['movie_id'].value_counts().to_dict()

        # Filter and score movies based on genre overlap and popularity
        recommendations = []
        for index, row in self.movies.iterrows():
            if row['movie_id'] in movie_popularity:
                overlap = self.calculate_genre_overlap(row['genres'], user_profile)
                if overlap > 0:  # Only consider movies with at least some genre overlap
                    popularity_score = movie_popularity[row['movie_id']]
                    recommendations.append((row['title'], overlap, popularity_score))

        # Debugging: Print sample recommendations to verify structure and types
        print("Sample recommendations:", recommendations[:5])

        # Sort movies first by genre overlap then by popularity
        try:
            recommendations.sort(key=lambda x: (-x[1], -x[2]))
        except Exception as e:
            print("Error in sorting recommendations:", e)
            return []

        # Return the top 10 recommendations
        top_recommendations = recommendations[:10]
        return [title for title, overlap, popularity in top_recommendations]


def main():
    recommender = MovieRecommender()
    recommender.load_data()
    user_id = input("Enter User ID: ")
    try:
        user_ratings = recommender.get_user_ratings(user_id)
        user_profile = recommender.create_user_profile(user_id)
        recommendations = recommender.recommend_movies(user_id)
        print("Top 10 Movie Recommendations:", recommendations)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
