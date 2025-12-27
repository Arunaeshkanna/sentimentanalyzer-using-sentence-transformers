import streamlit as st
from sentiment_engine import analyze_input_review

st.set_page_config(
    page_title="💬 AI Sentiment Analyzer (RAG)",
    page_icon="💡",
    layout="centered"
)

st.title("💬 AI-Powered Sentiment Analyzer")
st.caption("RAG using Sentence-Transformers (Stable Version)")

review = st.text_area(
    "📝 Write your review:",
    height=150,
    placeholder="E.g., I loved the product!"
)

if st.button("🔍 Analyze Sentiment"):
    if review.strip():
        with st.spinner("Analyzing..."):
            result = analyze_input_review(review)

        st.subheader("🎯 Predicted Sentiment")
        st.success(result["predicted_sentiment"].upper())

        st.subheader("🧠 Retrieved Similar Reviews")
        for r in result["similar_reviews"]:
            st.markdown(f"**{r['label'].capitalize()}**: {r['review_text']}")
    else:
        st.warning("Please enter a review")
