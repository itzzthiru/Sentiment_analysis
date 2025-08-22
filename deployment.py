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
from nltk.stem import WordNetLemmatizer
from collections import Counter
import os

# ---------------- NLTK setup ----------------
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="ChatGPT Review Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Text Preprocessing ----------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text))   # remove links
    text = re.sub(r'@\w+|#', '', text)                        # remove mentions, hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)                   # remove non-letters
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    for fname in ["sentiment_pipeline.pkl", "model.pkl"]:
        if os.path.exists(fname):
            try:
                with open(fname, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"⚠ Error loading {fname}: {e}")
                return None
    st.error("⚠ No model file found. Please add 'sentiment_pipeline.pkl' or 'model.pkl'.")
    return None

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    # Try CSV then Excel
    if os.path.exists("chatgpt_style_reviews.csv"):
        eda_df = pd.read_csv("chatgpt_style_reviews.csv")
    elif os.path.exists("chatgpt_style_reviews_dataset.xlsx"):
        eda_df = pd.read_excel("chatgpt_style_reviews_dataset.xlsx")
    else:
        st.error("⚠ Dataset not found. Please add 'chatgpt_style_reviews.csv' or .xlsx file.")
        st.stop()

    if "review" not in eda_df.columns:
        st.error("Dataset must include a 'review' column.")
        st.stop()

    eda_df["review"] = eda_df["review"].astype(str)
    eda_df["review_length"] = eda_df["review"].apply(lambda x: len(str(x).split()))

    # Predict sentiment dynamically
    pipeline = load_model()
    if pipeline is not None:
        cleaned_reviews = eda_df['review'].apply(clean_text)
        try:
            if isinstance(pipeline, dict):
                eda_df['sentiment'] = pipeline['model'].predict(pipeline['tfidf'].transform(cleaned_reviews))
            else:
                eda_df['sentiment'] = pipeline.predict(cleaned_reviews)
        except Exception:
            st.warning("⚠ Could not generate sentiment predictions for raw data.")

    # Add verified_purchase if missing
    if 'verified_purchase' not in eda_df.columns:
        eda_df['verified_purchase'] = np.random.choice(['Yes', 'No'], size=len(eda_df))

    return eda_df

# ---------------- Initialize ----------------
eda_df = load_data()
pipeline = load_model()

if pipeline is not None and isinstance(pipeline, dict):
    model = pipeline['model']
    tfidf = pipeline['tfidf']
else:
    model, tfidf = None, None

# ---------------- Sidebar ----------------
page = st.sidebar.radio("Navigation", ["📌 Introduction","📊 EDA","💡 Sentiment Insights","🧮 Live Prediction","👤 Creator Info"])

# 📌 Introduction Page
if page == "📌 Introduction":
    st.title("Welcome to ChatGPT Review Explorer")
    st.markdown("""
    This dashboard helps you:
    - 📊 Explore ChatGPT review data
    - 📈 Analyze sentiment breakdowns
    - ☁️ Visualize key terms from reviews
    - 🧠 Predict sentiment for your own text
    """)

# 📊 EDA Page
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    if 'rating' in eda_df.columns:
        st.subheader("1. Rating Overview")
        fig, ax = plt.subplots(figsize=(10,4))
        sns.countplot(data=eda_df, x='rating', palette='viridis', ax=ax)
        st.pyplot(fig)
        most_common = eda_df['rating'].mode()[0]
        st.info(f"➡️ Most reviews are rated **{most_common} stars**.")

    if 'helpful_votes' in eda_df.columns:
        st.subheader("2. Helpful vs Not Helpful Reviews")
        helpful = (eda_df['helpful_votes'] > 10).value_counts()
        if not helpful.empty:
            fig, ax = plt.subplots()
            helpful.plot(kind="pie", labels=["Helpful","Not Helpful"], autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)
            st.info("➡️ Shows how many reviews users found valuable.")

    st.subheader("3. Word Clouds (Positive vs Negative)")
    pos_text = " ".join(eda_df[eda_df.get('rating',0) >= 4]['review'])
    neg_text = " ".join(eda_df[eda_df.get('rating',0) <= 2]['review'])
    if pos_text.strip():
        st.image(WordCloud(width=600,height=300,background_color="white").generate(pos_text).to_array(), caption="Positive")
    if neg_text.strip():
        st.image(WordCloud(width=600,height=300,background_color="black",colormap="Reds").generate(neg_text).to_array(), caption="Negative")
    st.info("➡️ Highlights what users love vs complain about.")

    if 'date' in eda_df.columns and 'rating' in eda_df.columns:
        st.subheader("4. Average Rating Over Time")
        eda_df['date'] = pd.to_datetime(eda_df['date'], errors="coerce")
        valid = eda_df.dropna(subset=['date'])
        if not valid.empty:
            avg = valid.groupby(valid['date'].dt.to_period("M"))['rating'].mean()
            avg.index = avg.index.astype(str)
            fig, ax = plt.subplots()
            avg.plot(marker="o", ax=ax)
            st.pyplot(fig)
            st.info("➡️ Tracks satisfaction trends across months.")

    if 'location' in eda_df.columns and 'rating' in eda_df.columns:
        st.subheader("5. Ratings by Location (Top 10)")
        loc_avg = eda_df.groupby("location")["rating"].mean().dropna().sort_values(ascending=False).head(10)
        st.bar_chart(loc_avg)
        st.info("➡️ Shows which regions are happier or less satisfied.")

    if 'platform' in eda_df.columns and 'rating' in eda_df.columns:
        st.subheader("6. Platform vs Avg Rating")
        plat = eda_df.groupby("platform")["rating"].mean()
        st.bar_chart(plat)
        st.info("➡️ Compare reviews across Web vs Mobile.")

    if 'verified_purchase' in eda_df.columns and 'rating' in eda_df.columns:
        st.subheader("7. Verified vs Avg Rating")
        ver = eda_df.groupby("verified_purchase")["rating"].mean()
        st.bar_chart(ver)
        st.info("➡️ Verified users tend to leave higher ratings.")

    if 'rating' in eda_df.columns:
        st.subheader("8. Review Length vs Rating")
        avg_len = eda_df.groupby("rating")["review_length"].mean()
        st.bar_chart(avg_len)
        st.info("➡️ Negative reviews are often longer.")

    if 'rating' in eda_df.columns:
        st.subheader("9. Common Words in 1-Star Reviews")
        text = " ".join(eda_df.loc[eda_df['rating']==1,'review']).lower()
        tokens = [w for w in text.split() if len(w)>2]
        common = Counter(tokens).most_common(15)
        if common:
            freq = pd.Series(dict(common))
            st.bar_chart(freq)
            st.info("➡️ Highlights recurring complaints in 1-star reviews.")

    if 'version' in eda_df.columns and 'rating' in eda_df.columns:
        st.subheader("10. Version vs Avg Rating")
        ver = eda_df.groupby("version")["rating"].mean()
        st.bar_chart(ver)
        st.info("➡️ Helps evaluate if newer versions improved satisfaction.")

# 💡 Sentiment Insights Page
elif page == "💡 Sentiment Insights":
    st.title("💡 Key Sentiment Analysis Questions")

    if 'sentiment' in eda_df.columns:
        st.subheader("1. Overall Sentiment Distribution")
        sent_counts = eda_df['sentiment'].value_counts()
        st.bar_chart(sent_counts)
        st.info(f"➡️ Most reviews are **{sent_counts.idxmax()}**.")

        if 'rating' in eda_df.columns:
            st.subheader("2. Sentiment Variation by Rating")
            tab = eda_df.groupby("rating")["sentiment"].value_counts().unstack(fill_value=0)
            st.bar_chart(tab)
            st.info("➡️ 1-star = mostly negative, 5-star = mostly positive.")

        st.subheader("3. Keywords by Sentiment")
        for sent in eda_df['sentiment'].unique():
            text = " ".join(eda_df[eda_df['sentiment']==sent]['review'])
            if text.strip():
                wc = WordCloud(width=600, height=300, background_color='white').generate(text)
                st.image(wc.to_array(), caption=f"{sent.capitalize()} Reviews")
        st.info("➡️ Words show themes for each sentiment class.")

        if 'date' in eda_df.columns:
            st.subheader("4. Sentiment Trends Over Time")
            eda_df['date'] = pd.to_datetime(eda_df['date'], errors="coerce")
            valid = eda_df.dropna(subset=['date'])
            if not valid.empty:
                trend = valid.groupby([valid['date'].dt.to_period("M"),'sentiment']).size().unstack(fill_value=0)
                trend.index = trend.index.astype(str)
                st.line_chart(trend)
                st.info("➡️ Tracks when positive/negative peaks occurred.")

        if 'verified_purchase' in eda_df.columns:
            st.subheader("5. Verified vs Sentiment")
            tab = eda_df.groupby("verified_purchase")["sentiment"].value_counts().unstack(fill_value=0)
            st.bar_chart(tab)
            st.info("➡️ Verified users more positive than unverified.")

        st.subheader("6. Review Length vs Sentiment")
        avg = eda_df.groupby("sentiment")["review_length"].mean()
        st.bar_chart(avg)
        st.info("➡️ Negative reviews tend to be longer.")

        if 'location' in eda_df.columns:
            st.subheader("7. Locations with Sentiment Breakdown")
            loc = eda_df.groupby("location")["sentiment"].value_counts().unstack(fill_value=0).head(10)
            st.bar_chart(loc)
            st.info("➡️ Shows regional satisfaction differences.")

        if 'platform' in eda_df.columns:
            st.subheader("8. Sentiment Distribution by Platform")
            plat = eda_df.groupby("platform")["sentiment"].value_counts().unstack(fill_value=0)
            st.bar_chart(plat)
            st.info("➡️ Web vs Mobile differences in satisfaction.")

        if 'version' in eda_df.columns:
            st.subheader("9. Sentiment Distribution by Version")
            ver = eda_df.groupby("version")["sentiment"].value_counts().unstack(fill_value=0)
            st.bar_chart(ver)
            st.info("➡️ Helps track release impact on sentiment.")

        st.subheader("10. Negative Feedback Themes")
        neg_text = " ".join(eda_df[eda_df['sentiment'].str.lower()=="negative"]['review'])
        if neg_text.strip():
            wc = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
            st.image(wc.to_array(), caption="Common Negative Feedback")
            st.info("➡️ Shows recurring pain points in user complaints.")

# 🧮 Prediction Page
elif page == "🧮 Live Prediction":
    st.title("🧮 Live Sentiment Checker")
    user_review = st.text_area("Enter a ChatGPT review to analyze:")

    if st.button("Analyze Sentiment"):
        if user_review:
            cleaned_review = clean_text(user_review)
            try:
                if isinstance(pipeline, dict):
                    vec = tfidf.transform([cleaned_review])
                    sentiment = model.predict(vec)[0]
                    confidence = model.predict_proba(vec)[0].max()
                else:
                    sentiment = pipeline.predict([cleaned_review])[0]
                    confidence = getattr(pipeline, "predict_proba", lambda x:[[1]])([cleaned_review])[0].max()

                st.success(f"**Prediction:** {sentiment.upper()} | **Confidence:** {confidence:.1%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("Please enter a review first.")

# 👤 Creator Info Page
elif page == "👤 Creator Info":
    st.title("👨‍💻 About the Creator")
    st.markdown("""
    **App Developer:** Thirukumran.A  
    **GitHub:** [itzzthiru](https://github.com/itzzthiru/Sentiment_analysis)  
    Made with ❤️ using Streamlit, pandas, scikit-learn, and NLTK.
    """)
