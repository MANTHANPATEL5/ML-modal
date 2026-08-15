import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from io import BytesIO


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="TF-IDF Tag Cloud",
    page_icon="☁️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("☁️ TF-IDF Tag Cloud")
st.write("Enter multiple documents and generate a TF-IDF based Tag Cloud.")


# ==========================================
# USER INPUT
# ==========================================

text = st.text_area(
    "Enter your text",
    height=250,
    placeholder="""Enter each document on a separate line.

Example:
Python is a programming language
Python is used for machine learning
Machine learning is useful for data analysis
Python is popular for data science"""
)


# ==========================================
# GENERATE BUTTON
# ==========================================

if st.button("☁️ Generate Tag Cloud", use_container_width=True):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text.")

    else:

        # ==========================================
        # SPLIT TEXT INTO DOCUMENTS
        # ==========================================

        documents = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

        if len(documents) < 2:

            st.warning(
                "⚠️ Please enter at least 2 lines of text."
            )

        else:

            try:

                # ==========================================
                # TF-IDF
                # ==========================================

                vectorizer = TfidfVectorizer(
                    stop_words="english"
                )

                tfidf_matrix = vectorizer.fit_transform(documents)

                # Get words
                words = vectorizer.get_feature_names_out()

                # Calculate total TF-IDF score
                scores = tfidf_matrix.sum(axis=0).A1

                # Create word-score dictionary
                word_scores = dict(
                    zip(words, scores)
                )

                # ==========================================
                # CREATE WORD CLOUD
                # ==========================================

                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color="white",
                    min_font_size=10
                ).generate_from_frequencies(
                    word_scores
                )

                # ==========================================
                # DISPLAY TAG CLOUD
                # ==========================================

                st.success("✅ Tag Cloud generated successfully!")

                st.subheader("☁️ Generated TF-IDF Tag Cloud")

                fig, ax = plt.subplots(
                    figsize=(14, 7)
                )

                ax.imshow(
                    wordcloud,
                    interpolation="bilinear"
                )

                ax.axis("off")

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                # ==========================================
                # SAVE IMAGE TO MEMORY
                # ==========================================

                image_buffer = BytesIO()

                fig.savefig(
                    image_buffer,
                    format="png",
                    bbox_inches="tight",
                    dpi=200
                )

                image_buffer.seek(0)

                plt.close(fig)

                # ==========================================
                # DOWNLOAD BUTTON
                # ==========================================

                st.download_button(
                    label="⬇️ Download Tag Cloud",
                    data=image_buffer,
                    file_name="tfidf_tag_cloud.png",
                    mime="image/png",
                    use_container_width=True
                )

                # ==========================================
                # TOP TF-IDF WORDS
                # ==========================================

                st.subheader("📊 Top TF-IDF Words")

                sorted_scores = sorted(
                    word_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for word, score in sorted_scores[:10]:

                    st.write(
                        f"**{word}** : `{score:.4f}`"
                    )

            except ValueError:

                st.error(
                    "❌ Not enough valid words to generate the Tag Cloud."
                )