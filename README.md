Machine Learning Practice Projects

A collection of beginner-to-intermediate machine learning projects completed as part of the Ellevvo Learning Program.
Each project focuses on applying core ML concepts such as regression, classification, clustering, and recommendation systems using real-world datasets.

📚 Table of Contents

Overview

Projects

1. Student Performance Prediction

2. Mall Customer Segmentation

3. Loan Approval Prediction

4. Movie Recommendation System

Tools and Libraries

How to Run

Acknowledgement

🧩 Overview

This repository contains four key projects that explore different aspects of machine learning.
Each project was designed to build practical understanding in data preprocessing, model training, evaluation, and visualization.
The datasets were obtained from Kaggle and related open-source repositories.

💻 Projects
1. Student Performance Prediction

Goal: Predict students’ exam scores based on study hours and other performance factors.
Techniques: Linear Regression
Dataset: StudentPerformanceFactors.csv
Key Tasks:

Data cleaning and visualization

Splitting into training and testing sets

Training a linear regression model

Predicting and visualizing exam performance

📁 Script: Student_Performance.py

2. Mall Customer Segmentation

Goal: Cluster mall customers into segments based on annual income and spending score.
Techniques: K-Means Clustering
Dataset: Mall_Customers.csv
Key Tasks:

Data scaling and preprocessing

Determining optimal number of clusters (Elbow Method)

Applying K-Means

Visualizing customer segments

📁 Script: Mall_Segmentation.py

3. Loan Approval Prediction

Goal: Predict whether a loan application will be approved based on applicant details.
Techniques: Logistic Regression / Random Forest Classifier
Dataset: loan_approval_dataset.csv
Key Tasks:

Handling missing values and categorical encoding

Model training and evaluation

Dealing with imbalanced data

Measuring precision, recall, and F1-score

📁 Script: Loan_Approval_Prediction.py

4. Movie Recommendation System

Goal: Recommend top-rated movies to users based on their similarity to others.
Techniques: User-Based Collaborative Filtering (Cosine Similarity)
Dataset: Top_10000_Movies.csv
Key Tasks:

Building a user-item rating matrix

Computing user similarity

Generating movie recommendations for a given user

Evaluating recommendation precision

📁 Script: Movie_Recommender.py

🛠️ Tools and Libraries

Python 3.10+

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

🚀 How to Run

Clone this repository:

git clone  https://github.com/ahmedML2001/ML-PROJECTS-.git
cd ML-PROJECTS


Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn


Run any of the scripts:

python Student_Performance.py
python Mall_Segmentation.py
python Loan_Approval_Prediction.py
python Movie_Recommender.py

🙌 Acknowledgement

These projects were completed as part of the Ellevvo Learning Program, an initiative to enhance technical and practical data science skills through guided, real-world machine learning exercises.