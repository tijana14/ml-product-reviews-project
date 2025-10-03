# ml-product-reviews-project
A complete project for classifying product review sentiment

# ml-product-reviews-project
A complete project for classifying product review sentiment

🧠Sentiment analysis ML project (Complete pipeline)

This repository contains machine learning pipeline for sentiment analysis of product reviews using Python and scikit-learn.

🤝Project structure

|---- data
| `---product_reviews_full.csv # dataset
|----notebooks/
| `---exploratory_analysis.ipynb # EDA and preprocessing
|----src/
| `---train_model.py #script for training and saving the model
| `---test_model.py #script for testing saved model
`----README.md

✅What we did in this module

Throughout this module, we covered all major steps of a real world ML project.

1. Project setup
- created a new GitHub repository
- defined project folder structure
- uploaded raw dataset

2. Data exploration
- loaded and analyzed a large dataset with product reviewes
- used matplotlib and seaborn for visualization
- invastigated distribution of sentiments and text characteristics

3. Data cleaning and preprocessing
- removed missing values
- standardized sentiment labels (positive/negaitve/neutral)
- parsed nad validated prices
- converted review text to numerical length

4. Feature engineering
- selected meaningful input features: review_title, review_text and review_length
- removed irrelevant columns
- explored correlation between price and sentiment

5. Model training and evaluation
- compared multiple ML models (logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM)
- used ColumnTransformer and Pipeline for unified preprocessing
- evaluated using precision, recall, F1-score, and confusion matrix

6. Final model training
- trained final model on full dataset
- saved the pipeline using joblib to sentiment_model.pkl

7. Inference and usage
- loaded saved model
- built an interactive interface for predicting sentiment of new reviews
- enable real-time testing via console input

🚀How to use

🔧Train the model

 cd src
 python train_model.py

 This will created a file called sentiment_model.pkl in the root directory.

 🔎Run interface

 Use an interactive script (model_text.py) to classify new reviews using the trained model.

 ✨Author

 Tijana Dajic

 📜Licence
 
 This project is open-source and freely available for use.


