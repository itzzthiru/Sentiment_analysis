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

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')

# Page Config
st.set_page_config(
    page_title="ChatGPT Review Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\\S+|www\\S+|https\\S+", '', str(text))
    text = re.sub(r'\\@w+|\\#','', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load Model
@st.cache_resource
def load_model():
    with open('sentiment_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

# ✅ Load both datasets
@st.cache_data
def load_data():
    # Original dataset for EDA and Sentiment Insights
    eda_df = pd.read_csv("chatgpt_style_reviews.csv")
    eda_df['review_length'] = eda_df['review'].apply(lambda x: len(str(x).split()))

    # Predict sentiment dynamically
    cleaned_reviews = eda_df['review'].apply(clean_text)
    pipeline = load_model()
    eda_df['sentiment'] = pipeline['model'].predict(pipeline['tfidf'].transform(cleaned_reviews))

    # Synthetic verified_purchase if missing
    if 'verified_purchase' not in eda_df.columns:
        eda_df['verified_purchase'] = np.random.choice(['Yes', 'No'], size=len(eda_df))

    # Balanced dataset for prediction
    try:
        balance_df = pd.read_csv("final_data_with_sentiment.csv")
    except FileNotFoundError:
        balance_df = pd.read_csv("final_data.csv")
        balance_df['review_length'] = balance_df['review'].apply(lambda x: len(str(x).split()))
        cleaned_reviews_bal = balance_df['review'].apply(clean_text)
        pipeline = load_model()
        balance_df['sentiment'] = pipeline['model'].predict(pipeline['tfidf'].transform(cleaned_reviews_bal))
        balance_df.to_csv("final_data_with_sentiment.csv", index=False)

    return eda_df, balance_df

# Initialize
eda_df, df = load_data()
pipeline = load_model()
model = pipeline['model']
tfidf = pipeline['tfidf']

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["📌 Introduction","📊 EDA","🧠 Sentiment Insights","🧠 Live Prediction","👤 Creator Info"])

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

    # 1️⃣ Rating Distribution
    if 'rating' in eda_df.columns:
        st.subheader("1. Rating Overview")
        fig1, ax1 = plt.subplots(figsize=(10,4))
        sns.countplot(data=eda_df, x='rating', palette='viridis', ax=ax1)
        st.pyplot(fig1)

    # 2️⃣ Sentiment Pie + Review Length Box
    st.subheader("2. Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots()
        eda_df['sentiment'].value_counts().plot.pie(
            autopct='%1.1f%%',
            colors=['#4CAF50', '#FFC107', '#F44336'],
            ax=ax2
        )
        ax2.set_ylabel("")
        st.pyplot(fig2)

    with col2:
        fig3, ax3 = plt.subplots()
        sns.boxplot(
            data=eda_df,
            x='sentiment',
            y='review_length',
            order=['positive', 'neutral', 'negative'],
            palette=['#4CAF50', '#FFC107', '#F44336']
        )
        ax3.set_yscale('log')
        st.pyplot(fig3)

    # 3️⃣ Word Clouds (Rating-Based)
    st.subheader("3. Word Clouds (Based on Ratings)")
    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        st.subheader("Positive Reviews (4-5 Stars)")
        positive_text = " ".join(eda_df[eda_df['rating'] >= 4]['review'])
        if positive_text.strip():
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(positive_text)
            plt.figure(figsize=(10,5))
            plt.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("ℹ️ No positive (4-5 star) reviews available.")

    with wc_col2:
        st.subheader("Negative Reviews (1 Star Only)")
        negative_text = " ".join(eda_df[eda_df['rating'] == 1]['review'])
        if negative_text.strip():
            wordcloud = WordCloud(width=600, height=300, background_color='black', colormap='Reds').generate(negative_text)
            plt.figure(figsize=(10,5))
            plt.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("ℹ️ No 1-star reviews available.")

    # 4️⃣ Platform Insights
    if 'platform' in eda_df.columns:
        st.subheader("4. Platform Insights")
        platform_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        sns.countplot(data=eda_df, x='platform', ax=ax1)
        ax1.set_title("Reviews by Platform")
        sns.boxplot(data=eda_df, x='platform', y='rating', ax=ax2)
        ax2.set_title("Rating Distribution by Platform")
        st.pyplot(platform_fig)

# 🧠 Sentiment Insights Page
elif page == "🧠 Sentiment Insights":
    st.title("🧠 Key Sentiment Analysis Questions")

    # 1️⃣ Overall Sentiment
    st.subheader("1. Overall Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    eda_df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#4CAF50','#FFC107','#F44336'], ax=ax1)
    ax1.set_ylabel("")
    st.pyplot(fig1)

    # 2️⃣ Sentiment vs Rating
    st.subheader("2. Sentiment Variation by Rating")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.countplot(x='rating', hue='sentiment', data=eda_df, palette='Set2', ax=ax2)
    st.pyplot(fig2)

    # 3️⃣ Keywords by Rating Category
    st.subheader("3. Keywords by Rating Category")

    pos_text = " ".join(eda_df[eda_df['rating'] >= 4]['review'])
    if pos_text.strip():
        wc_pos = WordCloud(width=600, height=300, background_color='white').generate(pos_text)
        st.image(wc_pos.to_array(), caption="Positive Reviews (4-5 Stars)")

    neu_text = " ".join(eda_df[eda_df['rating'] == 3]['review'])
    if neu_text.strip():
        wc_neu = WordCloud(width=600, height=300, background_color='lightgray').generate(neu_text)
        st.image(wc_neu.to_array(), caption="Neutral Reviews (3 Stars)")

    neg_text = " ".join(eda_df[eda_df['rating'] == 1]['review'])
    if neg_text.strip():
        wc_neg = WordCloud(width=600, height=300, background_color='black', colormap='Reds').generate(neg_text)
        st.image(wc_neg.to_array(), caption="Negative Reviews (1 Star Only)")

    # 4️⃣ Sentiment Over Time
    if 'date' in eda_df.columns:
        st.subheader("4. Sentiment Trends Over Time")
        eda_df['date'] = pd.to_datetime(eda_df['date'], errors='coerce', format='mixed')
        eda_df.dropna(subset=['date'], inplace=True)
        sentiment_time = eda_df.groupby(['date','sentiment']).size().unstack(fill_value=0)
        fig4, ax4 = plt.subplots(figsize=(10,4))
        sentiment_time.plot(ax=ax4)
        st.pyplot(fig4)

    # 5️⃣ Verified vs Sentiment
    st.subheader("5. Sentiment Distribution: Verified vs Non-Verified")
    fig5, ax5 = plt.subplots(figsize=(8,4))
    sns.countplot(x='verified_purchase', hue='sentiment', data=eda_df, palette='coolwarm', ax=ax5)
    ax5.set_xlabel("Verified Purchase")
    ax5.set_ylabel("Number of Reviews")
    st.pyplot(fig5)

    # 6️⃣ Review Length vs Sentiment
    st.subheader("6. Review Length by Sentiment")
    fig6, ax6 = plt.subplots(figsize=(8,4))
    sns.boxplot(x='sentiment', y='review_length', data=eda_df, palette='autumn', ax=ax6, order=['negative','neutral','positive'])
    ax6.set_yscale('log')
    st.pyplot(fig6)

    # 7️⃣ Locations with Most Positive & Negative Sentiment
    if 'location' in eda_df.columns:
        st.subheader("7. Locations with Most Positive & Negative Sentiment")
        loc_sent = eda_df.groupby(['location', 'sentiment']).size().unstack(fill_value=0)
        top_locs = loc_sent.sum(axis=1).nlargest(10).index
        fig7, ax7 = plt.subplots(figsize=(10, 5))
        loc_sent.loc[top_locs].plot(kind='bar', stacked=True, ax=ax7, colormap='coolwarm')
        st.pyplot(fig7)

    # 8️⃣ Platform vs Sentiment
    if 'platform' in eda_df.columns:
        st.subheader("8. Sentiment Distribution by Platform")
        fig8, ax8 = plt.subplots(figsize=(6,4))
        sns.countplot(x='platform', hue='sentiment', data=eda_df, palette='pastel', ax=ax8)
        st.pyplot(fig8)

    # 9️⃣ Version vs Sentiment
    if 'version' in eda_df.columns:
        st.subheader("9. Sentiment Distribution by ChatGPT Version")
        fig_ver, ax_ver = plt.subplots(figsize=(8,4))
        sns.countplot(x='version', hue='sentiment', data=eda_df, palette='crest', ax=ax_ver)
        plt.xticks(rotation=45)
        st.pyplot(fig_ver)

    # 🔟 Negative Feedback Themes
    st.subheader("10. Common Negative Feedback Themes")
    neg_text = " ".join(eda_df[eda_df['rating'] == 1]['review'])
    if neg_text.strip():
        wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
        st.image(wc_neg.to_array(), caption="Negative Feedback Themes (1 Star Only)")
    else:
        st.info("ℹ️ No 1-star reviews available.")

# 🧠 Prediction Page
elif page == "🧠 Live Prediction":
    st.title("🧠 Live Sentiment Checker")
    user_review = st.text_area("Enter a ChatGPT review to analyze:")

    if st.button("Analyze Sentiment"):
        if user_review:
            cleaned_review = clean_text(user_review)
            vec = tfidf.transform([cleaned_review])
            sentiment = model.predict(vec)[0]
            confidence = model.predict_proba(vec)[0].max()

            st.success(f"""
            **Prediction:** {sentiment.upper()}  
            **Confidence:** {confidence:.1%}  
            **Processed Text:** {cleaned_review[:200]}...
            """)

            if sentiment == 'positive':
                st.balloons()
            elif sentiment == 'negative':
                st.warning("⚠️ This review contains negative sentiment.")
        else:
            st.error("Please enter a review first.")


# 👤 Creator Info Page
elif page == "👤 Creator Info":
    st.title("👨‍💻 About the Creator")
    st.markdown("""
    **App Developer:** Thirukumran.A ** 
    **GitHub:** https://github.com/itzzthiru/Sentiment_analysis **
     Made with ❤️ using Streamlit, pandas, scikit-learn, and NLTK.
    """)
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", use_container_width=True)