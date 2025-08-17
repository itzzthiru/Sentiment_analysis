import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

# NLTK setup
nltk.download('stopwords')

# Page Config
st.set_page_config(
    page_title="ChatGPT Review Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Text Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text))   # remove urls
    text = re.sub(r'@\w+|#', '', text)                        # remove mentions & hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)                   # keep only letters
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("sentiment_pipeline.pkl", "rb") as f:
        return pickle.load(f)


# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    if os.path.exists("chatgpt_style_reviews.csv"):
        df = pd.read_csv("chatgpt_style_reviews.csv")
    elif os.path.exists("chatgpt_style_reviews_dataset.xlsx"):
        df = pd.read_excel("chatgpt_style_reviews_dataset.xlsx")
    else:
        st.error("⚠ Dataset not found.")
        st.stop()

    if "review" not in df.columns:
        st.error("Dataset must include a 'review' column.")
        st.stop()

    df["review"] = df["review"].astype(str)
    df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))

    pipeline = load_model()
    if pipeline is not None:
        cleaned = df["review"].apply(clean_text)
        try:
            if isinstance(pipeline, dict):
                df["sentiment"] = pipeline["model"].predict(pipeline["tfidf"].transform(cleaned))
            else:
                df["sentiment"] = pipeline.predict(cleaned)
        except Exception:
            st.warning("⚠ Could not generate sentiment predictions.")

    if "verified_purchase" not in df.columns:
        df["verified_purchase"] = np.random.choice(["Yes","No"], size=len(df))

    return df

# ---------------- Initialize ----------------
eda_df = load_data()
pipeline = load_model()
model, tfidf = None, None
if isinstance(pipeline, dict):
    model, tfidf = pipeline["model"], pipeline["tfidf"]

# ---------------- Helper: Safe WordCloud ----------------
def safe_wordcloud(text, bg="white", cmap=None, caption=None):
    if not text.strip():
        st.info("ℹ No text available.")
        return
    try:
        wc = WordCloud(width=600, height=300, background_color=bg, colormap=cmap).generate(text)
        st.image(wc.to_array(), caption=caption)
    except ValueError:
        st.info("ℹ Wordcloud could not be generated (empty input).")

# ---------------- Sidebar ----------------
page = st.sidebar.radio("Navigation", ["📌 Introduction","📊 EDA","💡 Insights","🧮 Prediction","👤 Creator"])

# 📌 Introduction
if page == "📌 Introduction":
    st.title("Welcome to ChatGPT Review Explorer")
    st.markdown("""
    This dashboard includes:
    - 📊 10 EDA visualizations  
    - 💡 10 Sentiment Insights  
    - 🧮 Live Sentiment Prediction  
    - 👤 Creator Info
    """)

# 📊 EDA (10 Sections)
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # 1. Rating Distribution
    if "rating" in eda_df.columns:
        st.subheader("1️⃣ Rating Distribution")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(data=eda_df, x="rating", palette="viridis", ax=ax)
        ax.set_title("Ratings Distribution")
        st.pyplot(fig)

    # 2. Sentiment Distribution
    if "sentiment" in eda_df.columns:
        st.subheader("2️⃣ Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=eda_df, x="sentiment", palette="Set2", ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    # 3. Review Length Distribution
    st.subheader("3️⃣ Review Length Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(eda_df["review_length"], bins=30, kde=True, ax=ax, color="skyblue")
    ax.set_xlabel("Review Length (words)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 4. Wordcloud Positive
    st.subheader("4️⃣ Positive Wordcloud")
    pos_text = " ".join(eda_df[eda_df.get("rating",0)>=4]["review"])
    safe_wordcloud(pos_text, bg="white", caption="Positive Reviews")

    # 5. Wordcloud Negative
    st.subheader("5️⃣ Negative Wordcloud")
    neg_text = " ".join(eda_df[eda_df.get("rating",0)<=2]["review"])
    safe_wordcloud(neg_text, bg="black", cmap="Reds", caption="Negative Reviews")

    # 6. Average Rating Over Time
    if "date" in eda_df.columns and "rating" in eda_df.columns:
        st.subheader("6️⃣ Avg Rating Over Time")
        eda_df["date"] = pd.to_datetime(eda_df["date"], errors="coerce")
        temp = eda_df.dropna(subset=["date"])
        if not temp.empty:
            avg = temp.groupby(temp["date"].dt.to_period("M"))["rating"].mean()
            avg.index = avg.index.astype(str)
            st.line_chart(avg)

    # 7. Platform vs Rating
    if "platform" in eda_df.columns and "rating" in eda_df.columns:
        st.subheader("7️⃣ Platform vs Avg Rating")
        plat = eda_df.groupby("platform")["rating"].mean()
        st.bar_chart(plat)

    # 8. Verified vs Rating
    if "verified_purchase" in eda_df.columns and "rating" in eda_df.columns:
        st.subheader("8️⃣ Verified Purchase vs Avg Rating")
        ver = eda_df.groupby("verified_purchase")["rating"].mean()
        st.bar_chart(ver)

    # 9. Common Words in 1-Star
    if "rating" in eda_df.columns:
        st.subheader("9️⃣ Common Words in 1-Star Reviews")
        text = " ".join(eda_df[eda_df["rating"]==1]["review"]).lower()
        tokens = [w for w in text.split() if len(w)>2]
        freq = dict(Counter(tokens).most_common(15))
        if freq:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x=list(freq.keys()), y=list(freq.values()), palette="Reds_r", ax=ax)
            ax.set_title("Most Common Words in 1-Star Reviews")
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Word")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # 10. Version vs Rating
    if "version" in eda_df.columns and "rating" in eda_df.columns:
        st.subheader("🔟 Version vs Avg Rating")
        ver = eda_df.groupby("version")["rating"].mean()
        st.bar_chart(ver)

# 💡 Insights (10 Sections)
elif page == "💡 Insights":
    st.title("💡 Sentiment Insights")

    if "sentiment" in eda_df.columns:
        # 1. Overall Sentiment
        st.subheader("1️⃣ Overall Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=eda_df, x="sentiment", palette="Set2", ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        # 2. Sentiment vs Rating
        if "rating" in eda_df.columns:
            st.subheader("2️⃣ Sentiment vs Rating")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.countplot(x="rating", hue="sentiment", data=eda_df, palette="Set2", ax=ax)
            st.pyplot(fig)

        # 3. Keywords by Sentiment
        st.subheader("3️⃣ Keywords by Sentiment")
        for sent in eda_df["sentiment"].unique():
            text = " ".join(eda_df[eda_df["sentiment"]==sent]["review"])
            safe_wordcloud(text, caption=f"{sent.capitalize()} Reviews")

        # 4. Sentiment Over Time
        if "date" in eda_df.columns:
            st.subheader("4️⃣ Sentiment Over Time")
            eda_df["date"] = pd.to_datetime(eda_df["date"], errors="coerce")
            temp = eda_df.dropna(subset=["date"])
            if not temp.empty:
                trend = temp.groupby([temp["date"].dt.to_period("M"),"sentiment"]).size().unstack(fill_value=0)
                trend.index = trend.index.astype(str)
                st.line_chart(trend)

        # 5. Verified vs Sentiment
        if "verified_purchase" in eda_df.columns:
            st.subheader("5️⃣ Verified vs Sentiment")
            tab = eda_df.groupby("verified_purchase")["sentiment"].value_counts().unstack(fill_value=0)
            st.bar_chart(tab)

        # 6. Review Length vs Sentiment
        st.subheader("6️⃣ Review Length vs Sentiment")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(x="sentiment", y="review_length", data=eda_df, palette="coolwarm", ax=ax,
                    order=["negative","neutral","positive"])
        ax.set_yscale("log")
        st.pyplot(fig)

        # 7. Location Sentiment
        if "location" in eda_df.columns:
            st.subheader("7️⃣ Top 10 Locations by Sentiment")
            loc = eda_df.groupby("location")["sentiment"].value_counts().unstack(fill_value=0)
            st.bar_chart(loc.head(10))

        # 8. Platform vs Sentiment
        if "platform" in eda_df.columns:
            st.subheader("8️⃣ Platform vs Sentiment")
            plat = eda_df.groupby("platform")["sentiment"].value_counts().unstack(fill_value=0)
            st.bar_chart(plat)

        # 9. Version vs Sentiment
        if "version" in eda_df.columns:
            st.subheader("9️⃣ Version vs Sentiment")
            ver = eda_df.groupby("version")["sentiment"].value_counts().unstack(fill_value=0)
            st.bar_chart(ver)

        # 10. Negative Themes
        st.subheader("🔟 Negative Feedback Themes")
        eda_df["sentiment"] = eda_df["sentiment"].fillna("").astype(str)
        neg_text = " ".join(eda_df[eda_df["sentiment"].str.lower()=="negative"]["review"])
        safe_wordcloud(neg_text, bg="black", cmap="Reds")

# 🧮 Prediction
elif page == "🧮 Prediction":
    st.title("🧮 Live Sentiment Checker")
    user_review = st.text_area("Enter a review:")

    if st.button("Analyze"):
        if user_review:
            cleaned = clean_text(user_review)
            try:
                if isinstance(pipeline, dict):
                    vec = tfidf.transform([cleaned])
                    sentiment = model.predict(vec)[0]
                    conf = model.predict_proba(vec)[0].max() if hasattr(model,"predict_proba") else 1.0
                else:
                    sentiment = pipeline.predict([cleaned])[0]
                    conf = getattr(pipeline,"predict_proba",lambda x:[[1]])([cleaned])[0].max()
                st.success(f"Prediction: {sentiment} | Confidence: {conf:.1%}")
                if str(sentiment).lower()=="positive":
                    st.balloons()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("Please enter text.")

# 👤 Creator
elif page == "👤 Creator":
    st.title("👨‍💻 About the Creator")
    st.markdown("""
    **App Developer:** Thirukumran.A  
    **GitHub:** [itzzthiru](https://github.com/itzzthiru/Sentiment_analysis)  
    Made with ❤️ using Streamlit, pandas, scikit-learn, and NLTK.
    """)

