import streamlit as st
import pickle

# page config
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# load model
with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# load vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# title
st.title("🎬 IMDb Sentiment Analysis App")

st.write(
    "Analyze movie reviews using Machine Learning "
    "(Logistic Regression + TF-IDF)"
)

# sidebar
st.sidebar.header("About")

st.sidebar.write(
    """
    This project compares:
    - Logistic Regression
    - SVM
    - Random Forest
    - LSTM
    
    Built using:
    - Scikit-learn
    - TensorFlow
    - Streamlit
    """
)

# initialize session state
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# example reviews
st.subheader("Example Reviews")

col1, col2 = st.columns(2)

with col1:
    if st.button("Positive Example"):
        st.session_state.review_text = (
            "This movie was amazing with brilliant acting and emotional storytelling."
        )

with col2:
    if st.button("Negative Example"):
        st.session_state.review_text = (
            "This was the worst movie I have ever watched. Completely boring."
        )

# text input
review = st.text_area(
    "Enter a movie review",
    value=st.session_state.review_text,
    height=150
)

# prediction
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        review_tfidf = tfidf.transform([review])

        prediction = model.predict(review_tfidf)

        probability = model.predict_proba(review_tfidf)

        confidence = max(probability[0]) * 100

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("😊 Positive Review")

        else:
            st.error("😔 Negative Review")

        st.write(f"### Confidence: {confidence:.2f}%")