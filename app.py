import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

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
            # Load objects
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))
            poster_url = []
            # Map suggestion indices (shape (1, n_neighbors)) to book names, then poster urls
            for book_id in suggestion[0]:
                book_name = book_pivot.index[book_id]
                match = np.where(final_rating['title'] == book_name)[0]
                if len(match) > 0:
                    idx = match[0]
                    url = final_rating.iloc[idx]['image_url']
                    poster_url.append(url)
            return poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            model = pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            # Get integer index for book_name
            book_id_arr = np.where(book_pivot.index == book_name)[0]
            if len(book_id_arr) == 0:
                raise Exception("Book not found in pivot index.")
            book_id = int(book_id_arr[0])
            # n_neighbors=11 (for: 1 input + 10 recs)
            dist, suggestion = model.kneighbors(book_pivot.iloc[book_id].values.reshape(1, -1), n_neighbors=11)
            books_list = [book_pivot.index[book_id] for book_id in suggestion[0]]
            poster_url = self.fetch_poster(suggestion)
            # skip the selected book itself at index 0; return top 10
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
            st.subheader("🔟 Recommended Books For You:")
            cols = st.columns(10)
            for idx in range(10):
                with cols[idx]:
                    st.text(recommended_books[idx])
                    st.image(poster_url[idx])
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            logging.error(f"Recommendation Error: {e}")
            raise AppException(e, sys) from e

def main():
    st.header('📚 End to End Books Recommender System')
    st.write("Collaborative filtering based recommendation system!")

    obj = Recommendation()

    # Train Model Button
    if st.button('Train Recommender System'):
        obj.train_engine()

    try:
        # Load book names for dropdown
        book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))
        selected_book = st.selectbox(
            "Type or select a book from the dropdown",
            book_names
        )
        if st.button('Show Recommendation'):
            obj.recommendations_engine(selected_book)
    except Exception as e:
        st.error(f"Could not load books or generate recommendations: {e}")

if __name__ == "__main__":
    main()
