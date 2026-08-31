import streamlit as st
from pypdf import PdfReader
import re

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="centered"
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
# CHECK PDF
# ==========================================

if uploaded_file is None:

    st.info("📄 Please upload a PDF to start.")

    st.stop()


# ==========================================
# READ PDF
# ==========================================

try:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

except Exception as e:

    st.error(f"❌ Error reading PDF: {e}")

    st.stop()


# ==========================================
# CHECK EXTRACTED TEXT
# ==========================================

if not text.strip():

    st.error(
        "❌ No readable text was found in this PDF."
    )

    st.stop()


# ==========================================
# PDF SUCCESS MESSAGE
# ==========================================

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


# ==========================================
# PROCESS QUESTION
# ==========================================

if question:

    # ======================================
    # EXIT
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
        r"\b\w+\b",
        question.lower()
    )


    # ======================================
    # STOP WORDS
    # ======================================

    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "can",
        "does",
        "do"
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
        r"(?<=[.!?])\s+",
        text.replace("\n", " ")
    )


    # ======================================
    # SEARCH PDF
    # ======================================

    results = []


    for sentence in sentences:

        sentence_lower = sentence.lower()

        score = 0

        for word in question_words:

            if word in sentence_lower:

                score += 1


        if score > 0:

            results.append(
                (score, sentence.strip())
            )


    # ======================================
    # SORT RESULTS
    # ======================================

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )


    # ======================================
    # CREATE ANSWER
    # ======================================

    if results:

        answer_sentences = []

        for score, sentence in results[:3]:

            if sentence not in answer_sentences:

                answer_sentences.append(sentence)


        answer = " ".join(
            answer_sentences
        )

    else:

        answer = (
            "Sorry, I could not find the answer "
            "in the uploaded PDF."
        )


    # ======================================
    # SAVE ANSWER
    # ======================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


    # ======================================
    # REFRESH
    # ======================================

    st.rerun()
