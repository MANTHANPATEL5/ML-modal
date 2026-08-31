import streamlit as st
from pypdf import PdfReader
import re

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖"
)

st.title("🤖 PDF Question Answer Chatbot")

st.write(
    "Upload a PDF and ask questions based on its content."
)

# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)

# ==========================================
# LOAD PDF
# ==========================================

if uploaded_file is not None:

    try:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        # ==========================================
        # CHECK TEXT
        # ==========================================

        if not text.strip():

            st.error(
                "❌ No readable text was found in this PDF."
            )

            st.stop()

        st.success(
            f"✅ PDF uploaded successfully! "
            f"Pages: {len(reader.pages)}"
        )

        # ==========================================
        # CHAT HISTORY
        # ==========================================

        if "messages" not in st.session_state:

            st.session_state.messages = []

        # ==========================================
        # DISPLAY CHAT HISTORY
        # ==========================================

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.write(message["content"])

        # ==========================================
        # USER QUESTION
        # ==========================================

        question = st.chat_input(
            "Ask a question from the PDF..."
        )

        if question:

            # ======================================
            # EXIT COMMAND
            # ======================================

            if question.lower().strip() == "exit":

                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Goodbye! 👋"
                })

                st.rerun()

            # ======================================
            # SAVE USER QUESTION
            # ======================================

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            # ======================================
            # CLEAN QUESTION
            # ======================================

            question_words = re.findall(
                r'\b\w+\b',
                question.lower()
            )

            # Remove very common words
            stop_words = {
                "what", "is", "are", "the", "a",
                "an", "of", "to", "in", "on",
                "for", "and", "or", "how", "why",
                "when", "where", "which", "who",
                "can", "does", "do"
            }

            question_words = [
                word
                for word in question_words
                if len(word) > 2
                and word not in stop_words
            ]

            # ======================================
            # SPLIT PDF INTO SENTENCES
            # ======================================

            sentences = re.split(
                r'(?<=[.!?])\s+',
                text.replace("\n", " ")
            )

            # ======================================
            # FIND BEST ANSWER
            # ======================================

            best_sentences = []

            for sentence in sentences:

                sentence_lower = sentence.lower()

                score = 0

                for word in question_words:

                    if word in sentence_lower:
                        score += 1

                if score > 0:

                    best_sentences.append(
                        (score, sentence.strip())
                    )

            # ======================================
            # SORT RESULTS
            # ======================================

            best_sentences.sort(
                key=lambda x: x[0],
                reverse=True
            )

            # ======================================
            # RESPONSE
            # ======================================

            if best_sentences:

                # Take up to 3 relevant sentences
                selected_sentences = [
                    item[1]
                    for item in best_sentences[:3]
                ]

                answer = " ".join(
                    selected_sentences
                )

            else:

                answer = (
                    "Sorry, I could not find the answer "
                    "in the uploaded PDF."
                )

            # ======================================
            # SAVE RESPONSE
            # ======================================

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # ======================================
            # REFRESH
            # ======================================

            st.rerun()

else:

    st.info(
        "📄 Please upload a PDF to start asking questions."
    )
