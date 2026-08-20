import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="NLP Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# ==========================================================
# DOWNLOAD NLTK DATA
# ==========================================================

@st.cache_resource
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk_data()


# ==========================================================
# TITLE
# ==========================================================

st.title("📝 NLP Paragraph Summarizer")
st.write("Enter a paragraph and generate a short summary using TF-IDF.")


# ==========================================================
# USER INPUT
# ==========================================================

paragraph = st.text_area(
    "Enter your paragraph:",
    height=250,
    placeholder="Enter a long paragraph here..."
)


# ==========================================================
# SUMMARY BUTTON
# ==========================================================

if st.button("✨ Generate Summary", use_container_width=True):

    if not paragraph.strip():

        st.warning("⚠️ Please enter a paragraph.")

    else:

        # ==================================================
        # SENTENCE TOKENIZATION
        # ==================================================

        sentences = sent_tokenize(paragraph)

        # ==================================================
        # SHORT PARAGRAPH
        # ==================================================

        if len(sentences) <= 2:

            summary = paragraph.strip()

        else:

            # ==============================================
            # TF-IDF
            # ==============================================

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            tfidf_matrix = vectorizer.fit_transform(sentences)

            # ==============================================
            # SENTENCE SCORES
            # ==============================================

            sentence_scores = tfidf_matrix.sum(axis=1)

            scores = []

            for i in range(len(sentences)):
                scores.append(
                    float(sentence_scores[i, 0])
                )

            # ==============================================
            # SUMMARY LENGTH
            # ==============================================

            summary_length = max(
                1,
                len(sentences) // 2
            )

            # ==============================================
            # SELECT IMPORTANT SENTENCES
            # ==============================================

            top_sentences = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:summary_length]

            # ==============================================
            # KEEP ORIGINAL ORDER
            # ==============================================

            top_sentences = sorted(top_sentences)

            # ==============================================
            # CREATE SHORT SUMMARY
            # ==============================================

            summary = " ".join(
                sentences[i]
                for i in top_sentences
            )

        # ==================================================
        # SAVE SUMMARY
        # ==================================================

        st.session_state["summary"] = summary
        st.session_state["original_sentences"] = len(sentences)


# ==========================================================
# DISPLAY SHORT SUMMARY
# ==========================================================

if "summary" in st.session_state:

    summary = st.session_state["summary"]

    st.subheader("📌 Short Summary")

    # ======================================================
    # DISPLAY SUMMARY
    # ======================================================

    st.success(summary)

    # ======================================================
    # COPY BUTTON
    # ======================================================

    st.download_button(
        label="📋 Copy Summary",
        data=summary,
        file_name="summary.txt",
        mime="text/plain",
        use_container_width=False
    )

    # ======================================================
    # INFORMATION
    # ======================================================

    summary_sentences = len(
        sent_tokenize(summary)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"**Original Sentences:** "
            f"{st.session_state['original_sentences']}"
        )

    with col2:
        st.write(
            f"**Summary Sentences:** "
            f"{summary_sentences}"
        )