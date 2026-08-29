import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Message Spam Detection",
    page_icon="📧",
    layout="centered"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("📧 Message Spam Detection")

st.write(
    "Naive Bayes Spam Detection"
)


# ==========================================================
# LOAD DATASET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "ham_spam_dataset.csv"

df = pd.read_csv(DATA_PATH)


# ==========================================================
# X AND Y
# ==========================================================

x = df["Message"]
y = df["Label"]


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# COUNT VECTORIZER
# ==========================================================

vectorizer = CountVectorizer()

x_train_vectorized = vectorizer.fit_transform(x_train)

x_test_vectorized = vectorizer.transform(x_test)


# ==========================================================
# NAIVE BAYES MODEL
# ==========================================================

model = MultinomialNB()

model.fit(
    x_train_vectorized,
    y_train
)


# ==========================================================
# MODEL ACCURACY
# ==========================================================

y_pred = model.predict(
    x_test_vectorized
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==========================================================
# USER INPUT
# ==========================================================

st.subheader("📩 Enter Your Message")

message = st.text_area(
    "Message",
    value="",
    placeholder="Enter your message here..."
)


# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button(
    "🔮 Predict Message",
    use_container_width=True
):

    # ======================================================
    # VALIDATE INPUT
    # ======================================================

    if message.strip() == "":

        st.warning(
            "⚠️ Please enter a message."
        )

    else:

        # ==================================================
        # CONVERT MESSAGE TO NUMBERS
        # ==================================================

        message_vector = vectorizer.transform(
            [message]
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(
            message_vector
        )


        # ==================================================
        # PROBABILITY
        # ==================================================

        probability = model.predict_proba(
            message_vector
        )


        # ==================================================
        # GET CLASS PROBABILITIES
        # ==================================================

        ham_probability = 0
        spam_probability = 0

        for class_name, prob in zip(
            model.classes_,
            probability[0]
        ):

            class_name = str(class_name).strip().lower()

            if class_name == "ham":

                ham_probability = prob * 100

            elif class_name == "spam":

                spam_probability = prob * 100


        # ==================================================
        # PREDICTION RESULT
        # ==================================================

        st.subheader("🎯 Prediction Result")


        prediction_value = str(
            prediction[0]
        ).strip().lower()


        if prediction_value == "spam":

            st.error(
                "🚨 SPAM MESSAGE"
            )

        else:

            st.success(
                "✅ HAM / NOT SPAM"
            )


        # ==================================================
        # PROBABILITY
        # ==================================================

        st.subheader("📊 Prediction Probability")


        st.write(
            f"Ham: **{ham_probability:.2f}%**"
        )

        st.progress(
            min(100, max(0, int(ham_probability)))
        )


        st.write(
            f"Spam: **{spam_probability:.2f}%**"
        )

        st.progress(
            min(100, max(0, int(spam_probability)))
        )


        # ==================================================
        # MODEL ACCURACY
        # ==================================================

        st.divider()

        st.subheader("🎯 Model Accuracy")

        st.metric(
            "Naive Bayes Accuracy",
            f"{accuracy * 100:.2f}%"
        )
