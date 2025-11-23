import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException
from books_recommender.components.data_ingestion import DataIngestion

# ----- CSS Styling -----
st.markdown("""
    <style>
    .main {background-color: #f8f9fb;}
    .reportview-container .markdown-text-container {
        font-family: 'Open Sans', sans-serif;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0px 4px 12px #c1c6cc;
        padding: 32px;
        margin: 16px 0;
    }
    h1, h2 {color: #401cc4;}
    .stButton>button {
        border-radius: 5px;
        background-color: #3916a1;
        color: #fff;
        font-weight: bold;
        margin: 8px 0px;
        padding: 10px 26px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📚 Book Recommender ML System")
st.sidebar.markdown("""
*End-to-end Data Ingestion, Model Training, and Recommendations!*
""")

# ----- Navigation -----
menu = st.sidebar.radio("Navigate", ["🏁 Ingest Data", "📈 Train Model", "🤖 Book Recommendation", "ℹ️ About"])

# ----- DataIngestion Class -----
class DataIngestionUI:
    def run(self):
        st.header("🏁 Step 1: Data Ingestion")
        st.write("This downloads and prepares your book dataset for the recommender system.")
        run_ingest = st.button("Download & Extract Data")
        if run_ingest:
            with st.spinner("Running Data Ingestion..."):
                try:
                    di = DataIngestion(app_config=AppConfiguration())
                    ingested_dir = di.initiate_data_ingestion()
                    st.success("✅ Data successfully downloaded and extracted!")
                except Exception as e:
                    st.error(f"❌ Data ingestion failed: {e}")

# ----- Recommendation Class -----
class Recommendation:
    def __init__(self,app_config = AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects,'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]: 
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
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

            poster_url = self.fetch_poster(suggestion)
            for i in range(len(suggestion)):
                books = book_pivot.index[suggestion[i]]
                for j in books:
                    books_list.append(j)
            return books_list , poster_url   
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.success("Model Training Completed! 🚀")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = self.recommend_book(selected_books)
            st.subheader("📚 Top 5 Book Recommendations for you:")
            cols = st.columns(5)
            for idx in range(1,6):  # top 5 excluding first (the selected book itself)
                with cols[idx-1]:
                    st.text(recommended_books[idx])
                    st.image(poster_url[idx])
        except Exception as e:
            st.error("Error generating recommendations!")
            raise AppException(e, sys) from e

# --- UI Sections ---
if menu == "🏁 Ingest Data":
    DataIngestionUI().run()

elif menu == "📈 Train Model":
    st.header("📈 Step 2: Train Recommendation Model")
    st.write("Train a collaborative filtering ML model. This may take a few minutes depending on your dataset size.")
    recommender = Recommendation()
    if st.button("🛠️ Train Model"):
        with st.spinner("Training model..."):
            recommender.train_engine()

elif menu == "🤖 Book Recommendation":
    st.header("🤖 Step 3: Get Book Recommendations")
    try:
        # Load book names for the dropdown
        book_names_path = os.path.join("templates", "book_names.pkl")  # Adjust as per your project
        book_names = pickle.load(open(book_names_path , 'rb'))
        recommended = Recommendation()
        selected_book = st.selectbox(
            "Type or select a book you love:",
            book_names
        )
        if st.button("Show Recommendations"):
            recommended.recommendations_engine(selected_book)
    except Exception as e:
        st.error(f"⚠️ Unable to load recommendations UI: {e}")

elif menu == "ℹ️ About":
    st.title("ℹ️ About The Book Recommender App")
    st.markdown("""
    **End-to-End ML Book Recommender:**  
    - Data ingestion, model training, and real-time recommendations
    - Built for learning & production
    - Easily extensible for new recommendation features

    **Developed with ❤️ using Streamlit & Python**
    """)

# Footer / Contact
st.sidebar.markdown("---")
st.sidebar.info("🤍 Developed by an aspiring ML practitioner Kamran Sohail | Powered by Streamlit")
