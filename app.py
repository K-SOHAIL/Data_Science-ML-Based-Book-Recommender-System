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
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))

            # Collect book names from suggestion indices
            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            # Get indices of final_rating by title
            for name in book_name[0]:
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            # Get poster URLs
            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            # n_neighbors=11: 1 for the selected book, 10 recommendations
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=11)

            poster_url = self.fetch_poster(suggestion)
            # Return 10 recommendations, skip the first since it's the queried book itself
            for i in range(len(suggestion)):
                books = book_pivot.index[suggestion[i]]
                for j in books:
                    books_list.append(j)
            return books_list[1:11], poster_url[1:11]  # Take the top 10, skipping the selected book
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = self.recommend_book(selected_books)
            st.subheader("🔟 Recommended Books for You:")
            cols = st.columns(10)
            for idx in range(10):
                with cols[idx]:
                    st.text(recommended_books[idx])
                    st.image(poster_url[idx])
        except Exception as e:
            st.error("Error generating recommendations!")
            raise AppException(e, sys) from e

if __name__ == "__main__":
    st.header('End to End Books Recommender System')
    st.text("This is a collaborative filtering based recommendation system!")

    obj = Recommendation()

    # Training
    if st.button('Train Recommender System'):
        obj.train_engine()

    book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))
    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        book_names
    )

    # Show 10 recommendations
    if st.button('Show Recommendation'):
        obj.recommendations_engine(selected_books)
        