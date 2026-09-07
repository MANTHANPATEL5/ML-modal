import streamlit as st
import joblib
import re


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🤖",
    layout="centered"
)


# ==========================================================
# LOAD SAVED MODEL PACKAGE
# ==========================================================

try:
    model_package = joblib.load("interview_model.pkl")

except FileNotFoundError:
    st.error("interview_model.pkl not found.")
    st.info("Run job_analyzer.py first.")
    st.stop()


# ==========================================================
# CHECK MODEL FORMAT
# ==========================================================

if isinstance(model_package, dict):

    vectorizer = model_package["vectorizer"]
    model = model_package["model"]

else:

    vectorizer = None
    model = model_package


# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("🤖 AI Interview Analyzer")

st.write(
    "Analyze a candidate using Resume, Interview Transcript "
    "and Job Description."
)


# ==========================================================
# CANDIDATE INFORMATION
# ==========================================================

st.subheader("👤 Candidate Information")

candidate_name = st.text_input(
    "Candidate Name",
    placeholder="Example: Jay Patel"
)

role = st.text_input(
    "Job Role",
    placeholder="Example: Software Engineer"
)


# ==========================================================
# CANDIDATE RESUME
# ==========================================================

st.subheader("📄 Candidate Resume")

resume = st.text_area(
    "Paste Resume",
    height=220,
    placeholder="Paste the candidate's resume here..."
)


# ==========================================================
# INTERVIEW TRANSCRIPT
# ==========================================================

st.subheader("🎤 Interview Transcript")

transcript = st.text_area(
    "Paste Interview Transcript",
    height=300,
    placeholder="Paste the interview transcript here..."
)


# ==========================================================
# JOB DESCRIPTION
# ==========================================================

st.subheader("💼 Job Description")

job_description = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder="Paste the job description here..."
)


# ==========================================================
# TEXT SIMILARITY FUNCTION
# ==========================================================

def calculate_similarity(text1, text2):

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not text1.strip() or not text2.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    vectors = vectorizer.fit_transform(
        [text1, text2]
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return similarity * 100


# ==========================================================
# INTERVIEW SCORE
# ==========================================================

def calculate_interview_score(transcript):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        transcript
    )

    word_count = len(words)

    if word_count == 0:
        return 0.0

    # Interview length component
    length_score = min(
        100,
        (word_count / 250) * 100
    )

    # Technical and behavioral indicators
    indicators = [
        "experience",
        "project",
        "developed",
        "implemented",
        "python",
        "machine learning",
        "sql",
        "problem",
        "solution",
        "team",
        "leadership",
        "communication",
        "testing",
        "debugging",
        "data",
        "analysis",
        "technology",
        "software",
        "development"
    ]

    text = transcript.lower()

    found = 0

    for word in indicators:

        if word in text:
            found += 1

    indicator_score = (
        found / len(indicators)
    ) * 100

    # Final interview score
    score = (
        length_score * 0.30
        +
        indicator_score * 0.70
    )

    return min(
        100,
        score
    )


# ==========================================================
# ANALYZE BUTTON
# ==========================================================

if st.button(
    "🔍 Analyze Candidate",
    use_container_width=True
):

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    if not candidate_name.strip():

        st.warning(
            "Please enter the candidate name."
        )
        st.stop()

    if not role.strip():

        st.warning(
            "Please enter the job role."
        )
        st.stop()

    if not resume.strip():

        st.warning(
            "Please enter the candidate resume."
        )
        st.stop()

    if not transcript.strip():

        st.warning(
            "Please enter the interview transcript."
        )
        st.stop()

    if not job_description.strip():

        st.warning(
            "Please enter the job description."
        )
        st.stop()


    # ======================================================
    # CREATE SAME TEXT FORMAT USED DURING TRAINING
    # ======================================================

    combined_text = (
        " RESUME_SECTION "
        + resume
        + " TRANSCRIPT_SECTION "
        + transcript
        + " JOB_SECTION "
        + job_description
    )


    # ======================================================
    # MACHINE LEARNING PREDICTION
    # ======================================================

    if vectorizer is not None:

        transformed_text = vectorizer.transform(
            [combined_text]
        )

        prediction = model.predict(
            transformed_text
        )[0]

    else:

        prediction = model.predict(
            [combined_text]
        )[0]


    # ======================================================
    # CONFIDENCE
    # ======================================================

    confidence = 0.0

    if vectorizer is not None:

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                transformed_text
            )[0]

            confidence = (
                max(probabilities) * 100
            )

        elif hasattr(
            model,
            "decision_function"
        ):

            decision_score = model.decision_function(
                transformed_text
            )

            score = abs(
                float(
                    decision_score[0]
                )
            )

            confidence = (
                50
                +
                (
                    50 *
                    (
                        1 -
                        (1 / (1 + score))
                    )
                )
            )

            confidence = min(
                100,
                confidence
            )

    else:

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                [combined_text]
            )[0]

            confidence = (
                max(probabilities) * 100
            )


    # ======================================================
    # RESUME MATCH
    # ======================================================

    resume_match = calculate_similarity(
        resume,
        job_description
    )


    # ======================================================
    # INTERVIEW RELEVANCE
    # ======================================================

    interview_relevance = calculate_similarity(
        transcript,
        job_description
    )


    # ======================================================
    # INTERVIEW SCORE
    # ======================================================

    interview_score = calculate_interview_score(
        transcript
    )


    # ======================================================
    # FINAL SCORE
    # ======================================================

    final_score = (
        resume_match * 0.30
        +
        interview_relevance * 0.25
        +
        interview_score * 0.20
        +
        confidence * 0.25
    )

    final_score = min(
        100,
        max(
            0,
            final_score
        )
    )


    # ======================================================
    # INTERVIEW ANALYSIS
    # ======================================================

    st.subheader(
        "📊 Interview Analysis"
    )


    # ======================================================
    # AI RECOMMENDATION
    # ======================================================

    if prediction == "select":

        st.success(
            "✅ AI RECOMMENDATION: SELECT"
        )

    else:

        st.error(
            "❌ AI RECOMMENDATION: REJECT"
        )


    # ======================================================
    # SCORE CARDS
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Final Score",
            f"{final_score:.1f}%"
        )

    with col2:

        st.metric(
            "Resume Match",
            f"{resume_match:.1f}%"
        )

    with col3:

        st.metric(
            "Interview Score",
            f"{interview_score:.1f}%"
        )


    # ======================================================
    # SECOND ROW
    # ======================================================

    col4, col5 = st.columns(2)

    with col4:

        st.metric(
            "Interview Relevance",
            f"{interview_relevance:.1f}%"
        )

    with col5:

        st.metric(
            "ML Confidence",
            f"{confidence:.1f}%"
        )


    # ======================================================
    # MACHINE LEARNING RESULT
    # ======================================================

    st.subheader(
        "🤖 Machine Learning Result"
    )

    # Model name intentionally removed
    # Test accuracy intentionally removed

    st.write(
        f"**Prediction:** "
        f"{prediction.upper()}"
    )

    st.write(
        f"**Prediction Confidence:** "
        f"{confidence:.2f}%"
    )


    # ======================================================
    # CANDIDATE DETAILS
    # ======================================================

    st.subheader(
        "👤 Candidate Details"
    )

    st.write(
        f"**Name:** "
        f"{candidate_name}"
    )

    st.write(
        f"**Role:** "
        f"{role}"
    )


    # ======================================================
    # FINAL RECOMMENDATION
    # ======================================================

    st.subheader(
        "📌 Final Recommendation"
    )

    if final_score >= 75:

        recommendation = (
            "Strong Candidate"
        )

        st.success(
            f"**{recommendation}**"
        )

    elif final_score >= 60:

        recommendation = (
            "Potential Candidate"
        )

        st.warning(
            f"**{recommendation}**"
        )

    else:

        recommendation = (
            "Needs Further Review"
        )

        st.info(
            f"**{recommendation}**"
        )

    st.write(
        f"Overall Candidate Score: "
        f"**{final_score:.1f}%**"
    )


    # ======================================================
    # SCORE BREAKDOWN
    # ======================================================

    st.subheader(
        "📈 Score Breakdown"
    )

    # ------------------------------------------------------
    # NO st.progress()
    # ------------------------------------------------------
    # This removes the blue horizontal bars completely.

    st.write(
        f"Resume Match — "
        f"**{resume_match:.1f}%**"
    )

    st.write(
        f"Interview Relevance — "
        f"**{interview_relevance:.1f}%**"
    )

    st.write(
        f"Interview Score — "
        f"**{interview_score:.1f}%**"
    )

    st.write(
        f"ML Confidence — "
        f"**{confidence:.1f}%**"
    )