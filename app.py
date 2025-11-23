import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

# --- Recommendation Engine ---
class Recommendation:
    def __init__(self, app_config=None):
        try:
            if app_config is None:
                app_config = AppConfiguration()
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))
            # suggestion: shape (1, n_neighbors). Get book names.
            book_names_list = [book_pivot.index[book_id] for book_id in suggestion[0]]
            ids_index = []
            poster_url = []
            for name in book_names_list:
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)
            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)
            return poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            model = pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            # Find correct book_id (make sure it's an int, not array)
            book_id_arr = np.where(book_pivot.index == book_name)[0]
            if len(book_id_arr) == 0:
                raise Exception("Book name not found in data.")
            book_id = int(book_id_arr[0])
            # Get 11 neighbors (1 is self, next 10 are recommendations)
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=11)
            poster_url = self.fetch_poster(suggestion)
            books_list = [book_pivot.index[book_id] for book_id in suggestion[0]]
            # Remove the selected book itself (first entry)
            return books_list[1:11], poster_url[1:11]
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.success("Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            st.error(f"Training Error: {e}")
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_book):
        try:
            recommended_books, poster_url = self.recommend_book(selected_book)
            st.subheader("🔟 Recommended Books for You:")
            cols = st.columns(10)
            for idx in range(10):
                with cols[idx]:
                    st.text(recommended_books[idx])
                    st.image(poster_url[idx])
        except Exception as e:
            st.error("Error generating recommendations!")
            logging.error(f"Recommendation Error: {e}")
            raise AppException(e, sys) from e

# --- Streamlit UI ---
def main():
    st.header('End to End Books Recommender System')
    st.write("This is a collaborative-filtering based recommendation system!")

    obj = Recommendation()

    # Model Training
    if st.button('Train Recommender System'):
        obj.train_engine()

    try:
        book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))
        selected_book = st.selectbox(
            "Type or select a book from the dropdown",
            book_names
        )
        if st.button('Show Recommendation'):
            obj.recommendations_engine(selected_book)
    except Exception as e:
        st.error(f"Could not load book choices or generate recommendations: {e}")

if __name__ == "__main__":
    main()
