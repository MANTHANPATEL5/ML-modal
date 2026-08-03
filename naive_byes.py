import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ==========================================
# PAGE
# ==========================================

st.title("📧 Message Spam Detection")


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("email_spam_dataset_10000_unique.csv")


# ==========================================
# X AND Y
# ==========================================

x = df["Message"]
y = df["Label"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# COUNT VECTORIZER
# ==========================================

vectorizer = CountVectorizer()

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)


# ==========================================
# NAIVE BAYES
# ==========================================

model = MultinomialNB()

model.fit(x_train, y_train)


# ==========================================
# MODEL ACCURACY
# ==========================================

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)


# ==========================================
# USER INPUT
# ==========================================

st.subheader("📩 Enter Your Message")

email = st.text_area(
    "Message",
    placeholder="Example: Congratulations! You won a free prize. Click now!"
)


# ==========================================
# PREDICT
# ==========================================

if st.button("Predict Message"):

    if email.strip() == "":
        st.warning("Please enter an email message.")

    else:

        # Convert email to numbers
        email_vector = vectorizer.transform([email])

        # Prediction
        prediction = model.predict(email_vector)

        # Probability
        probability = model.predict_proba(email_vector)

        # Get class names and probabilities
        classes = model.classes_

        ham_probability = 0
        spam_probability = 0

        for class_name, prob in zip(classes, probability[0]):

            if class_name == "Ham":
                ham_probability = prob * 100

            elif class_name == "Spam":
                spam_probability = prob * 100


        # ======================================
        # RESULT
        # ======================================

        if prediction[0] == "Spam":
            st.error("🚨 SPAM MESSAGE")

        else:
            st.success("✅ HAM / NOT SPAM")


        # ======================================
        # PROBABILITY
        # ======================================

        st.subheader("📊 Probability")

        st.write(
            f"Ham: **{ham_probability:.2f}%**"
        )

        st.progress(
            int(ham_probability)
        )

        st.write(
            f"Spam: **{spam_probability:.2f}%**"
        )

        st.progress(
            int(spam_probability)
        )


        # ======================================
        # ACCURACY
        # ======================================

        st.subheader("🎯 Model Accuracy")

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )
