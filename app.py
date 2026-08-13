
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 90% 0%, rgba(120, 70, 180, .14), transparent 32%),
        radial-gradient(circle at 0% 0%, rgba(220, 50, 70, .08), transparent 28%);
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background: #0d0d13;
    border-right: 1px solid rgba(255,255,255,.08);
}

.hero {
    padding: 2.4rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(105,55,180,.32), rgba(25,25,40,.92));
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 15px 45px rgba(0,0,0,.22);
    margin-bottom: 1.7rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    margin-top: .5rem;
    font-size: 1.08rem;
    opacity: .72;
    max-width: 760px;
    line-height: 1.6;
}

.feature-card, .movie-card {
    padding: 1.25rem;
    border-radius: 17px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.08);
}

.feature-card {
    min-height: 165px;
}

.movie-card {
    margin-bottom: .75rem;
}

.feature-title, .movie-title {
    font-weight: 700;
}

.feature-title {
    font-size: 1.1rem;
    margin-bottom: .55rem;
}

.feature-text {
    opacity: .67;
    line-height: 1.5;
    font-size: .92rem;
}

.movie-title {
    font-size: 1rem;
    margin-bottom: .5rem;
}

.movie-rank {
    font-size: .78rem;
    opacity: .45;
    margin-bottom: .25rem;
}

.score {
    display: inline-block;
    padding: .25rem .6rem;
    border-radius: 20px;
    background: rgba(255,193,7,.10);
    border: 1px solid rgba(255,193,7,.20);
    font-size: .82rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 15px;
    padding: 1rem;
}

.stButton > button {
    width: 100%;
    min-height: 44px;
    border-radius: 11px;
    font-weight: 650;
}

.footer {
    text-align: center;
    opacity: .42;
    font-size: .82rem;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,.07);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================

movies = pd.read_csv(
    "data/ml-100k/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)

ratings = pd.read_csv(
    "data/ml-100k/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# ============================================================
# POPULARITY
# ============================================================

popularity_df = (
    ratings.groupby("movie_id")
    .agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )
    .reset_index()
    .merge(movies, on="movie_id")
)

# ============================================================
# SVD
# ============================================================

@st.cache_resource
def train_svd():
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        ratings[["user_id", "movie_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    model = SVD(
        n_factors=20,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )

    model.fit(trainset)
    return model

svd_model = train_svd()

# ============================================================
# ITEM KNN
# ============================================================

@st.cache_resource
def build_item_knn():
    matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)

    similarity = cosine_similarity(matrix.T)

    return pd.DataFrame(
        similarity,
        index=matrix.columns,
        columns=matrix.columns
    )

movie_similarity_df = build_item_knn()

# ============================================================
# FUNCTIONS
# ============================================================

def search_movie(query):
    if not query:
        return pd.DataFrame()

    return movies[
        movies["title"].str.contains(
            query,
            case=False,
            na=False
        )
    ]


def cold_start_recommendations(n=10, min_ratings=50):
    result = (
        popularity_df[
            popularity_df["rating_count"] >= min_ratings
        ]
        .sort_values(
            ["average_rating", "rating_count"],
            ascending=False
        )
        .head(n)
        [["title", "average_rating", "rating_count"]]
        .copy()
    )

    result["average_rating"] = result["average_rating"].round(2)
    return result


def recommend_similar_movies(movie_title, n=10):
    row = movies[movies["title"] == movie_title]

    if row.empty:
        return pd.DataFrame()

    movie_id = row.iloc[0]["movie_id"]

    if movie_id not in movie_similarity_df.columns:
        return cold_start_recommendations(n)

    similarities = movie_similarity_df[movie_id]

    ids = (
        similarities
        .drop(movie_id)
        .sort_values(ascending=False)
        .head(n)
        .index
    )

    result = movies[movies["movie_id"].isin(ids)].copy()
    result["similarity"] = result["movie_id"].map(similarities)

    return result.sort_values(
        "similarity",
        ascending=False
    )


def personalized_recommendations(user_id, n=10):
    rated = set(
        ratings.loc[
            ratings["user_id"] == user_id,
            "movie_id"
        ]
    )

    candidates = movies[
        ~movies["movie_id"].isin(rated)
    ].copy()

    predictions = []

    for movie_id in candidates["movie_id"]:
        pred = svd_model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    pred_df = pd.DataFrame(
        predictions,
        columns=["movie_id", "predicted_rating"]
    )

    result = candidates.merge(
        pred_df,
        on="movie_id"
    )

    result = (
        result
        .sort_values("predicted_rating", ascending=False)
        .head(n)
        .copy()
    )

    result["predicted_rating"] = result["predicted_rating"].round(2)

    return result


def display_movie_cards(df, score_column, label):
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        st.markdown(
            f"""
            <div class="movie-card">
                <div class="movie-rank">#{rank}</div>
                <div class="movie-title">🎬 {row["title"]}</div>
                <span class="score">⭐ {label}: {row[score_column]}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_popular_cards(df):
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        st.markdown(
            f"""
            <div class="movie-card">
                <div class="movie-rank">#{rank}</div>
                <div class="movie-title">🎬 {row["title"]}</div>
                <span class="score">
                    ⭐ {row["average_rating"]}
                    &nbsp; • &nbsp;
                    👥 {row["rating_count"]} ratings
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 1.3rem;">
            <div style="font-size:3rem;">🎬</div>
            <h2 style="margin:0;">CineMatch</h2>
            <div style="opacity:.5;">ML Movie Recommender</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Recommendations",
            "📊 Model Performance",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.caption(
        "MovieLens 100K • Item-KNN • Matrix Factorization • SVD"
    )

# ============================================================
# HOME
# ============================================================

if page == "🏠 Recommendations":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🎬 CineMatch</div>
            <div class="hero-subtitle">
                Discover movies you'll love using machine learning.
                Find movies similar to your favorites or get
                personalized recommendations based on predicted ratings.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("👥 Users", "943")
    with c2:
        st.metric("🎬 Movies", "1,682")
    with c3:
        st.metric("⭐ Ratings", "100K")
    with c4:
        st.metric("🏆 Best RMSE", "0.9311")

    st.divider()

    st.subheader("🔎 Find a Movie")

    st.caption(
        "Search the MovieLens catalogue to find movies similar to a title."
    )

    movie_name = st.text_input(
        "Movie search",
        placeholder="Try: Toy Story, Star Wars, Titanic...",
        label_visibility="collapsed"
    )

    if movie_name:
        results = search_movie(movie_name)

        if results.empty:
            st.warning("😕 No movies found. Try another title.")
        else:
            selected_movie = st.selectbox(
                "Choose a movie",
                results["title"].tolist()
            )

            st.info(f"🎯 Selected: **{selected_movie}**")

            if st.button(
                "✨ Find Similar Movies",
                use_container_width=True
            ):
                recs = recommend_similar_movies(
                    selected_movie,
                    n=10
                )

                st.subheader("🍿 Movies You May Like")

                if "similarity" in recs.columns:
                    recs["similarity"] = recs["similarity"].round(3)
                    display_movie_cards(
                        recs,
                        "similarity",
                        "Similarity"
                    )
                else:
                    display_popular_cards(recs)

    st.divider()

    st.subheader("🎯 Recommendation Modes")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size:2rem;">🎬</div>
                <div class="feature-title">Similar Movies</div>
                <div class="feature-text">
                    Finds movies with similar user-rating patterns
                    using Item-Based Collaborative Filtering.
                    <br><br>
                    <b>Model:</b> Item-KNN<br>
                    <b>Similarity:</b> Cosine Similarity
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size:2rem;">👤</div>
                <div class="feature-title">Personalized Recommendations</div>
                <div class="feature-text">
                    Predicts ratings for movies a user has not rated
                    and returns the highest-scoring titles.
                    <br><br>
                    <b>Model:</b> Matrix Factorization / SVD
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("👤 Personalized Recommendations")

    st.caption(
        "Enter a MovieLens user ID. Known users receive SVD recommendations; "
        "unknown users receive a popularity-based cold-start fallback."
    )

    left, right = st.columns([1, 2])

    with left:
        user_id = st.number_input(
            "MovieLens User ID",
            min_value=1,
            max_value=9999,
            value=196,
            step=1
        )

        st.caption("MovieLens users: 1–943")

        get_recs = st.button(
            "🎯 Get My Recommendations",
            use_container_width=True
        )

    if get_recs:
        selected_user = int(user_id)

        if selected_user not in ratings["user_id"].unique():
            st.warning(
                "👤 New user detected. No rating history exists for this user."
            )

            st.subheader("🍿 Popular Movies to Get You Started")

            recs = cold_start_recommendations(10)
            display_popular_cards(recs)

        else:
            with st.spinner("🤖 Generating personalized recommendations..."):
                recs = personalized_recommendations(
                    selected_user,
                    n=10
                )

            st.success("✨ Recommendations generated successfully!")

            st.subheader("🍿 Recommended For You")

            display_movie_cards(
                recs,
                "predicted_rating",
                "Predicted Rating"
            )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📊 Model Performance</div>
            <div class="hero-subtitle">
                Evaluation results from the recommendation-system
                experiments, including rating prediction and ranking quality.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📁 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("👥 Users", "943")
    with c2:
        st.metric("🎬 Movies", "1,682")
    with c3:
        st.metric("⭐ Ratings", "100,000")
    with c4:
        st.metric("🕳️ Sparsity", "93.70%")

    st.divider()

    st.subheader("🤖 Models Explored")

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📈 Popularity Baseline</div>
                <div class="feature-text">
                    Recommends movies using rating quality and
                    sufficient rating history.
                </div>
            </div>
            <br>
            <div class="feature-card">
                <div class="feature-title">🎬 Item-Based Collaborative Filtering</div>
                <div class="feature-text">
                    Calculates movie-to-movie similarity from
                    user-rating patterns using cosine similarity.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🧮 Matrix Factorization</div>
                <div class="feature-text">
                    Implemented from scratch using latent user
                    and movie factors optimized with gradient descent.
                </div>
            </div>
            <br>
            <div class="feature-card">
                <div class="feature-title">🏆 Surprise SVD</div>
                <div class="feature-text">
                    Used as a benchmark against the from-scratch
                    matrix factorization implementation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("📈 Rating Prediction Performance")

    performance_df = pd.DataFrame({
        "Model": [
            "Scratch Matrix Factorization",
            "Surprise SVD"
        ],
        "Test RMSE": [
            0.9402,
            0.9311
        ]
    })

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🏆 Best Model", "Surprise SVD")
    with c2:
        st.metric("Best Test RMSE", "0.9311")
    with c3:
        st.metric("Improvement", "0.97%")

    st.bar_chart(
        performance_df.set_index("Model"),
        height=350
    )

    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Lower RMSE indicates better rating prediction accuracy."
    )

    st.divider()

    st.subheader("🎯 Ranking Performance")

    ranking_df = pd.DataFrame({
        "Metric": [
            "Precision@10",
            "Recall@10"
        ],
        "Score": [
            0.139,
            0.0567
        ]
    })

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Precision@10", "0.139")
        st.caption(
            "13.9% of Top-10 recommendations were relevant."
        )

    with c2:
        st.metric("Recall@10", "0.0567")
        st.caption(
            "5.67% of relevant test movies were retrieved."
        )

    st.bar_chart(
        ranking_df.set_index("Metric"),
        height=300
    )

    st.caption(
        "Evaluated on 100 users using a relevance threshold of 4.0."
    )

    st.divider()

    st.subheader("🧊 Cold-Start Handling")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size:2rem;">👤</div>
                <div class="feature-title">New User</div>
                <div class="feature-text">
                    Users without historical data are handled
                    with a popularity-based fallback using movies
                    with sufficient rating history.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size:2rem;">🎬</div>
                <div class="feature-title">Unknown Movie</div>
                <div class="feature-text">
                    Movies unavailable to the Item-KNN model
                    fall back to popularity-based recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">ℹ️ About CineMatch</div>
            <div class="hero-subtitle">
                A machine-learning project exploring collaborative
                filtering, matrix factorization, SVD and recommender
                system evaluation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🎬 Project Overview")

    st.write(
        """
        CineMatch is a movie recommendation system built using
        the MovieLens 100K dataset. The project focuses on the
        complete recommendation-system workflow: starting with
        simple baselines, implementing collaborative filtering
        and matrix factorization, tuning models, evaluating
        ranking quality and finally exposing the models through
        an interactive Streamlit application.
        """
    )

    st.divider()

    st.subheader("📁 Dataset")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Users", "943")
    with c2:
        st.metric("Movies", "1,682")
    with c3:
        st.metric("Ratings", "100,000")

    st.divider()

    st.subheader("🔬 Machine Learning Pipeline")

    pipeline = [
        ("01", "MovieLens", "Dataset"),
        ("02", "EDA", "Data analysis"),
        ("03", "Popularity Baseline", "Baseline"),
        ("04", "Item-KNN", "Collaborative filtering"),
        ("05", "Failure Analysis", "Analyze limitations"),
        ("06", "Matrix Factorization", "Implemented from scratch"),
        ("07", "Hyperparameter Tuning", "Model optimization"),
        ("08", "Surprise SVD", "Benchmark"),
        ("09", "Cold-Start", "Robustness"),
        ("10", "Ranking Evaluation", "Precision & Recall"),
        ("11", "Streamlit", "Interactive application")
    ]

    for number, title, description in pipeline:
        st.markdown(
            f"""
            <div class="movie-card">
                <div style="display:flex;gap:1rem;align-items:center;">
                    <div style="opacity:.4;min-width:30px;">
                        {number}
                    </div>
                    <div>
                        <div class="movie-title">{title}</div>
                        <div style="opacity:.58;font-size:.9rem;">
                            {description}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("🛠️ Technologies")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🐍 Python</div>
                <div class="feature-text">
                    Core programming language.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📊 ML Stack</div>
                <div class="feature-text">
                    NumPy · Pandas · Scikit-learn · Surprise
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🌐 Application</div>
                <div class="feature-text">
                    Streamlit interactive web application.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        """
        The objective was not simply to build a working
        recommendation interface. The project was designed
        to understand, implement, evaluate and compare
        multiple recommendation approaches and to handle
        practical issues such as sparse data and cold-start users.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🎬 CineMatch · Movie Recommendation System
        <br>
        Built with Python · Machine Learning · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
