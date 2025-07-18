import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text))
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load Model
@st.cache_resource
def load_model():
    with open('sentiment_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("final_data_with_sentiment.csv")
    except FileNotFoundError:
        df = pd.read_csv("final_data.csv")
        if 'review' not in df.columns:
            st.error("The CSV must contain a 'review' column.")
            st.stop()
        if 'review_length' not in df.columns:
            df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
        cleaned_reviews = df['review'].apply(clean_text)
        pipeline = load_model()
        df['sentiment'] = pipeline['model'].predict(pipeline['tfidf'].transform(cleaned_reviews))
        df.to_csv("final_data_with_sentiment.csv", index=False)
        st.success("✅ Sentiment predicted and saved as 'final_data_with_sentiment.csv'")
    return df

# Initialize
df = load_data()
pipeline = load_model()
model = pipeline['model']
tfidf = pipeline['tfidf']

# Sidebar navigation
page = st.sidebar.radio("Navigation", [
    "📌 Introduction",
    "📊 EDA",
    "🧠 Live Prediction",
    "👤 Creator Info"
])

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

    if 'rating' in df.columns:
        st.subheader("1. Rating Overview")
        fig1, ax1 = plt.subplots(figsize=(10,4))
        sns.countplot(data=df, x='rating', palette='viridis', ax=ax1)
        ax1.set_title("Distribution of Star Ratings")
        st.pyplot(fig1)

    st.subheader("2. Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots()
        df['sentiment'].value_counts().plot.pie(
            autopct='%1.1f%%',
            colors=['#4CAF50', '#FFC107', '#F44336'],
            ax=ax2
        )
        ax2.set_ylabel("")
        st.pyplot(fig2)

    with col2:
        fig3, ax3 = plt.subplots()
        sns.boxplot(
            data=df,
            x='sentiment',
            y='review_length',
            order=['positive', 'neutral', 'negative'],
            palette=['#4CAF50', '#FFC107', '#F44336']
        )
        ax3.set_yscale('log')
        st.pyplot(fig3)

    st.subheader("3. Word Clouds")
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.subheader("Positive Reviews")
        positive_text = " ".join(df[df['sentiment'] == 'positive']['review'])
        wordcloud = WordCloud(width=600, height=300, background_color='white').generate(positive_text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(plt)

    with wc_col2:
        st.subheader("Negative Reviews")
        negative_text = " ".join(df[df['sentiment'] == 'negative']['review'])
        wordcloud = WordCloud(width=600, height=300, background_color='black', colormap='Reds').generate(negative_text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(plt)

    if 'platform' in df.columns:
        st.subheader("4. Platform Insights")
        platform_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        sns.countplot(data=df, x='platform', ax=ax1)
        ax1.set_title("Reviews by Platform")
        sns.boxplot(data=df, x='platform', y='rating', ax=ax2)
        ax2.set_title("Rating Distribution by Platform")
        st.pyplot(platform_fig)

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
    **App Developer:** Your Name  
    **GitHub:** https://github.com/itzzthiru
     Made with ❤️ using Streamlit, pandas, scikit-learn, and NLTK.
    """)
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", use_container_width=True)

