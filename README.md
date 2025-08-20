# Sentiment Analysis of Movie Reviews

## Overview
This project aims to perform **sentiment analysis** on **movie reviews** from the entertainment industry. The task is to classify movie reviews into **positive** or **negative** categories, which has significant applications in **recommendation systems**, **marketing**, and **consumer insights**.

The project compares two types of models for sentiment classification:
1. **Traditional Statistical Model**: Naive Bayes classifier.
2. **Embedding-Based Model**: Logistic Regression using **Word2Vec** embeddings.

The models will be evaluated based on common **classification metrics** such as **accuracy**, **precision**, **recall**, and **F1-score** to determine their effectiveness in sentiment analysis.

## Objective
- **Text Classification**: Classifying movie reviews as either **positive** or **negative**.
- **Model Comparison**: Evaluating **Naive Bayes** and **Logistic Regression with Word2Vec** embeddings.
- **Performance Evaluation**: Using classification metrics (accuracy, precision, recall, F1-score) to evaluate model performance.

## Dataset
The dataset used in this project is a collection of **movie reviews** with labeled sentiments (**positive** or **negative**). The reviews are from **Rotten Tomatoes** or similar sources, and the goal is to train a model that accurately predicts sentiment from new reviews.

## Machine Learning Models Used
1. **Naive Bayes**: A simple, yet effective **statistical model** for text classification based on probability theory.
2. **Logistic Regression with Word2Vec Embeddings**: An **embedding-based model** that uses **Word2Vec** embeddings to capture semantic relationships between words in the movie reviews.

## Technologies Used
- **Python**: Programming language used for building and training the models.
- **Scikit-learn**: A Python library for machine learning that provides tools for building models like **Naive Bayes** and **Logistic Regression**.
- **Gensim**: Library for generating **Word2Vec** embeddings.
- **Pandas & NumPy**: Libraries for data manipulation and preprocessing.
- **Matplotlib & Seaborn**: Libraries for data visualization.

## Evaluation Metrics
- **Accuracy**: The proportion of correctly predicted instances.
- **Precision**: The proportion of true positive results in positive predictions.
- **Recall**: The proportion of true positives correctly identified out of actual positives.
- **F1-Score**: The harmonic mean of precision and recall.

