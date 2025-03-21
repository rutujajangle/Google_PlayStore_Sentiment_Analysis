# **Sentiment Analysis of Google Play Store Reviews using BERT**

## **Overview**
This project leverages the **BERT-base-uncased** model to enhance sentiment classification of Google Play Store reviews. By fine-tuning BERT on a curated dataset, it effectively classifies reviews into **Positive, Neutral, and Negative** categories. The project integrates **natural language processing (NLP) techniques**, **class balancing strategies**, and **transformer-based deep learning models** to improve classification accuracy and handle nuanced sentiments.

### **Key Features**
- **Preprocessing & Tokenization**: Uses **NLTK** for text preprocessing and **BERT tokenizer** for input representation.
- **Fine-tuned BERT Model**: Trained on **Google Play Store reviews** to adapt to domain-specific sentiment patterns.
- **Class Imbalance Handling**: Implements **weighted loss functions** to ensure fair learning across all sentiment classes.
- **High Classification Accuracy**: Achieves **94% accuracy and a macro F1-score of 92%**.
- **Visualization & Analysis**: Generates **training loss curves, confusion matrix, and classification reports** for performance evaluation.

---

## **Technologies Used**
- **Python** for scripting and data handling.
- **PyTorch & Transformers (Hugging Face)** for model fine-tuning.
- **NLTK & scikit-learn** for text preprocessing and evaluation.
- **Matplotlib & Seaborn** for visualizing sentiment distribution and model performance.

---

## **Directory Structure**
```bash
├── data/                          # Dataset and preprocessed reviews
├── models/                        # Saved fine-tuned BERT model
├── notebooks/                     # Jupyter notebooks for training and evaluation
├── src/                           # Scripts for preprocessing, training, and inference
├── visualizations/                # Plots for sentiment distribution and model performance
├── requirements.txt               # Required dependencies
└── README.md                      # Project documentation
```

---

## **Dataset**
The dataset consists of **Google Play Store reviews** labeled as **Positive, Neutral, and Negative** sentiments. It was sourced from:
- [Google Play Store Reviews Dataset](https://www.kaggle.com/code/gallo33henrique/llm-engineering-prompt-sentiment-analysis/input?select=googleplaystore_user_reviews.csv)

---

## **Models Used**
- **Pre-trained BERT-base-uncased** (fine-tuned on Play Store reviews for sentiment classification).

---

## **Results**
- **Accuracy**: 94%
- **Macro F1-Score**: 92%
- **Precision-Recall Breakdown**:
  | Sentiment  | Precision | Recall | F1-Score |
  |------------|-----------|--------|----------|
  | Negative   | 0.89      | 0.90   | 0.89     |
  | Neutral    | 0.92      | 0.92   | 0.92     |
  | Positive   | 0.96      | 0.96   | 0.96     |

- **Visualizations**:
  - **Training Loss Curve**
  - **Sentiment Distribution**
  - **Confusion Matrix**

---

## **Future Improvements**
- **Incorporate sarcasm detection** for better handling of implicit sentiments.
- **Experiment with larger transformer models** like **RoBERTa** or **DeBERTa**.
- **Extend dataset coverage** to include more diverse app categories.

---

## **License**
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---
