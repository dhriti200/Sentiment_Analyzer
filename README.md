# 🎬 IMDb Sentiment Analysis App

A Machine Learning + Deep Learning project for analyzing movie review sentiment using NLP techniques.

## 🚀 Live Demo

[Add your Streamlit app link here]

---

## 📌 Project Overview

This project compares multiple machine learning and deep learning models for sentiment classification on the IMDb movie reviews dataset.

The goal was to study:

* Classical ML vs Deep Learning
* Accuracy vs computational complexity
* NLP preprocessing pipelines
* Real-world deployment of ML models

---

## 🧠 Models Implemented

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 88.36%   |
| SVM                 | 87.02%   |
| Random Forest       | 84.00%   |
| LSTM                | 86.00%   |

---

## 🛠 Technologies Used

* Python
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## 📂 Project Structure

```bash
Sentiment_Analyzer/
│
├── app/
│   └── app.py
│
├── model/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── lstm_sentiment_model.h5
│
├── notebooks/
│   └── baseline.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Features

* Sentiment prediction for movie reviews
* Interactive Streamlit web app
* Confidence score display
* Comparison of multiple ML architectures
* Deep Learning implementation using LSTM
* Model evaluation and visualization

---

## 📊 Key Insights

* Logistic Regression outperformed more complex models on TF-IDF features.
* Random Forest struggled with sparse high-dimensional text vectors.
* LSTM achieved competitive performance but required higher computational cost.
* Classical ML models remain highly effective for sentiment analysis tasks.

---

## 🧪 Dataset

IMDb Movie Reviews Dataset

* 25,000 training reviews
* 25,000 testing reviews
* Binary sentiment classification (Positive / Negative)

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/dhriti200/Sentiment_Analyzer.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```

---

## 🌟 Future Improvements

* Add BERT transformer model
* Deploy using Docker
* Add model selection inside app
* Improve UI/UX
* Add sentiment confidence visualization

---

## 👩‍💻 Author

Dhriti
