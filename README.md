# 💬 Sentiment Analysis NLP Pipeline

**Author:** Manoj S Annigeri | [GitHub](https://github.com/Man27oj)

An end-to-end Natural Language Processing pipeline that classifies sentiment 
(positive, negative, neutral) from Twitter airline reviews using multiple 
ML approaches — from classical machine learning to state-of-the-art BERT transformers.

---

## 🎯 Project Overview

This project was built as part of my Master's application portfolio to demonstrate 
practical NLP and machine learning skills. It compares multiple modeling approaches 
on real-world messy social media data.

---

## 📊 Results

| Model | Accuracy |
|---|---|
| Naive Bayes (Baseline) | 73.26% |
| Logistic Regression + TF-IDF | 78.63% |
| LSTM Deep Learning | 76.37% |
| **BERT (Fine-tuned)** | **80.88%** |

**Key Finding:** BERT achieved the highest accuracy and significantly outperformed 
other models on nuanced negative sentences. LSTM underperformed on short tweets 
due to limited text length — demonstrating that model selection must consider 
data characteristics.

---

## 🗂️ Dataset

- **Source:** [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (Kaggle)
- **Size:** ~14,000 tweets
- **Classes:** Positive, Negative, Neutral
- **Challenge:** Highly imbalanced — 63% negative tweets

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Pandas, NumPy, NLTK |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Classical ML | Scikit-learn, TF-IDF |
| Deep Learning | TensorFlow, Keras, LSTM |
| Transformers | HuggingFace, BERT |
| Deployment | Streamlit |

---

## 🏗️ Project Structure
```
sentiment-analysis-nlp/
│
├── 01_data_exploration.ipynb    # EDA, cleaning, word clouds
├── 02_machine_learning.ipynb    # TF-IDF, Logistic Regression, Naive Bayes
├── 03_lstm_model.ipynb          # LSTM deep learning model
├── 04_bert_model.ipynb          # BERT fine-tuning
├── app.py                       # Streamlit live demo
└── README.md
```

---

## 🚀 How to Run Locally
```bash
# Clone the repo
git clone https://github.com/Man27oj/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 💡 Key Learnings

- Real-world tweet data is messy — cleaning pipeline is critical
- LSTM needs longer text to outperform classical ML; TF-IDF works well on short tweets
- BERT with only 1 epoch misclassified nuanced negatives; 3 epochs fixed this
- Model confidence scores reveal uncertainty that accuracy alone hides

---

## 📄 License

MIT License — feel free to use and build on this project!
