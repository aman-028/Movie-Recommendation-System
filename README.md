# 🎬 CineMatch — Movie Recommendation System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://movie-recommendation-system-ajkfdxvehe64cxaxkpd5ra.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end movie recommendation system built on the **MovieLens 100K dataset**, exploring popularity-based recommendation, Item-Based Collaborative Filtering, Matrix Factorization from scratch, hyperparameter tuning, Surprise SVD benchmarking, cold-start handling, ranking evaluation, and interactive deployment.

## 🌐 Live Demo

**[Open CineMatch](https://movie-recommendation-system-ajkfdxvehe64cxaxkpd5ra.streamlit.app/)**

## 📌 Overview

CineMatch was developed as a complete recommender-system study rather than simply using a pre-built recommendation library.

The project progressively moves from simple baselines to collaborative filtering and latent-factor models, evaluates their performance, investigates cold-start behavior, and exposes the final system through an interactive Streamlit application.

### Core recommendation modes

- 🎬 **Similar Movies** — Item-Based Collaborative Filtering using cosine similarity.
- 👤 **Personalized Recommendations** — predicted ratings from the SVD-based model.
- 🧊 **Cold-Start Fallback** — popularity-based recommendations for users without historical data or unsupported movie cases.

---

## 🔬 Machine Learning Pipeline

```text
MovieLens 100K
      ↓
EDA + Data Analysis
      ↓
Popularity Baseline
      ↓
Item-KNN
      ↓
Analyze Failures
      ↓
Matrix Factorization from Scratch
      ↓
Hyperparameter Tuning
      ↓
Surprise SVD Benchmark
      ↓
Cold-Start Experiment
      ↓
Ranking Evaluation
      ↓
Streamlit Application
      ↓
Public Deployment
```

---

## 📊 Dataset

The project uses the **MovieLens 100K** dataset.

| Statistic | Value |
|---|---:|
| Users | 943 |
| Movies | 1,682 |
| Ratings | 100,000 |
| Matrix sparsity | 93.70% |

The high sparsity of the user-movie matrix makes recommendation challenging because most users have rated only a small fraction of the available movies.

---

## 🤖 Models & Approaches

### 1. Popularity Baseline

A simple baseline that recommends movies using rating quality and sufficient rating history.

It also serves as the fallback strategy for cold-start situations.

### 2. Item-Based Collaborative Filtering

Movie-to-movie similarity is calculated from user-rating patterns.

**Method:**
- Construct user-movie rating matrix
- Fill missing ratings with zero for similarity computation
- Transpose to obtain movie vectors
- Calculate cosine similarity
- Retrieve the most similar movies

### 3. Matrix Factorization — From Scratch

A matrix-factorization model was implemented manually to understand the underlying recommendation algorithm.

The model learns:
- User latent factors
- Movie latent factors

The predicted rating is obtained from the interaction between the learned latent vectors.

Training uses gradient-based optimization over observed ratings.

### 4. Hyperparameter Tuning

The scratch matrix-factorization model was tested with different:
- Number of latent factors
- Learning rates
- Regularization values

The best configuration used for the final comparison was:

| Parameter | Value |
|---|---:|
| Latent factors | 20 |
| Learning rate | 0.005 |
| Regularization | 0.02 |

### 5. Surprise SVD

The Surprise library's SVD implementation was trained as a benchmark against the from-scratch matrix-factorization implementation.

Final test RMSE:

**0.9311**

---

## 📈 Experimental Results

### Rating Prediction

| Model | Test RMSE |
|---|---:|
| Matrix Factorization — From Scratch | **0.9402** |
| Surprise SVD | **0.9311** |

Lower RMSE indicates better rating-prediction accuracy.

Surprise SVD improved the test RMSE by approximately **0.97%** over the scratch implementation.

### Ranking Evaluation

The recommender was evaluated using:
- **Top-N = 10**
- **Relevant rating threshold = 4.0**
- **100 users evaluated**

| Metric | Score |
|---|---:|
| Precision@10 | **0.139** |
| Recall@10 | **0.0567** |

These metrics evaluate the quality of the recommendation ranking rather than only the accuracy of individual rating predictions.

---

## 🧊 Cold-Start Handling

Recommendation systems face a cold-start problem when there is insufficient historical information.

CineMatch handles two cases:

### New User

If a user is not present in the MovieLens dataset, the system falls back to popularity-based recommendations.

### Unknown / Unsupported Movie

If a movie cannot be handled by the Item-KNN similarity model, the system uses the popularity-based fallback.

This prevents the application from failing when collaborative information is unavailable.

---

## 🏗️ Application Architecture

```text
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       Similar Movies                    Personalized
              │                           Recommendations
              ↓                                 ↓
        Item-KNN +                         SVD Model
     Cosine Similarity                          │
              │                                 │
              └────────────────┬────────────────┘
                               ↓
                        Recommendation
                            Results
                               │
                               ↓
                         Streamlit UI
```

---

## 🛠️ Technology Stack

### Programming
- Python

### Data & Machine Learning
- NumPy
- Pandas
- Scikit-learn
- Scikit-Surprise

### Application
- Streamlit

### Development
- Jupyter Notebook
- Git
- GitHub

---

## 📁 Project Structure

```text
Movie-Recommendation-System/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── ml-100k/
│       └── ml-100k/
│           ├── u.data
│           ├── u.item
│           ├── u.user
│           ├── u.genre
│           └── ...
│
└── notebooks/
    ├── 01_EDA.ipynb
    └── 02_Collaborative_Filtering.ipynb
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scriptsctivate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the application

```powershell
streamlit run app.py
```

The application will open locally in your browser.

---

## 🎯 Project Objective

The main objective was to understand the recommendation-system workflow end to end:

1. Explore and understand the dataset.
2. Establish a simple popularity baseline.
3. Implement Item-Based Collaborative Filtering.
4. Analyze limitations and failures.
5. Implement Matrix Factorization from scratch.
6. Tune important hyperparameters.
7. Compare the scratch model against a standard SVD implementation.
8. Investigate cold-start behavior.
9. Evaluate recommendation ranking using Precision@10 and Recall@10.
10. Build and deploy an interactive application.

The project therefore combines **machine-learning implementation, experimentation, evaluation, and deployment** rather than relying only on a pre-trained or library-only recommender.

---

## 🔮 Future Improvements

Potential future extensions include:

- Movie posters and richer movie metadata
- Genre-aware recommendations
- Hybrid recommendation combining collaborative and content-based filtering
- Better cold-start strategies using movie metadata
- More extensive ranking evaluation such as NDCG@K and MAP@K
- Offline model persistence instead of training during application startup
- Faster batch prediction for personalized recommendations
- User rating interface for collecting new preferences
- Production API architecture with a dedicated backend
- Improved deployment and model-serving architecture

---

## 📚 Key Learning Outcomes

This project provided practical experience with:

- Sparse user-item matrices
- Collaborative filtering
- Cosine similarity
- Latent-factor models
- Gradient-based optimization
- Hyperparameter tuning
- RMSE evaluation
- Top-N recommendation evaluation
- Precision@K and Recall@K
- Cold-start problems
- Model benchmarking
- Streamlit application development
- Git/GitHub workflow
- Cloud deployment

---

## 👨‍💻 Project

**CineMatch — Movie Recommendation System**

Built as an end-to-end machine-learning recommendation-system project using MovieLens 100K.

**[🌐 Try the Live Application](https://movie-recommendation-system-ajkfdxvehe64cxaxkpd5ra.streamlit.app/)**
