import streamlit as st
from pypdf import PdfReader
from pathlib import Path

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖"
)

st.title("🤖 PDF Question Answer Chatbot")

st.write(
    "Ask questions based on the Machine Learning PDF."
)

# ==========================================
# PDF FILE PATH
# ==========================================

PDF_FILE = Path(__file__).parent / "Machine_Learning.pdf"

# ==========================================
# LOAD PDF ONLY ONE TIME
# ==========================================

@st.cache_data
def load_pdf():

    reader = PdfReader(str(PDF_FILE))

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================
# LOAD PDF
# ==========================================

try:

    text = load_pdf()

except FileNotFoundError:

    st.error(
        "❌ Machine_Learning.pdf was not found."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading PDF: {e}"
    )

    st.stop()


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
# USER INPUT
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
    # SHOW USER QUESTION
    # ======================================

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # ======================================
    # FIND ANSWER
    # ======================================

    question_words = question.lower().split()

    sentences = (
        text
        .replace("\n", " ")
        .split(".")
    )

    best_sentence = ""
    best_score = 0


    # ======================================
    # MATCH QUESTION WITH PDF
    # ======================================

    for sentence in sentences:

        sentence_lower = sentence.lower()

        score = 0

        for word in question_words:

            if len(word) > 2 and word in sentence_lower:

                score += 1


        if score > best_score:

            best_score = score
            best_sentence = sentence.strip()


    # ======================================
    # RESPONSE
    # ======================================

    if best_score > 0:

        answer = best_sentence

    else:

        answer = (
            "Sorry, I could not find the answer "
            "in the PDF."
        )


    # ======================================
    # SAVE ASSISTANT RESPONSE
    # ======================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


    # ======================================
    # REFRESH
    # ======================================

    st.rerun()
