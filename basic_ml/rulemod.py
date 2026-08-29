import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# Sample Transaction Data
# -----------------------------
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diaper', 'Beer', 'Eggs'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk', 'Diaper', 'Cola']
]

# -----------------------------
# One-Hot Encoding
# -----------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# -----------------------------
# Apriori Algorithm
# -----------------------------
frequent_itemsets = apriori(
    df,
    min_support=0.2,
    use_colnames=True
)

# -----------------------------
# Association Rules
# -----------------------------
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Product Recommendation", layout="centered")

st.title("🛒 Product Recommendation System")

product = st.text_input("Enter Product Name")

if st.button("Search"):

    if product.strip() == "":
        st.warning("Please enter a product name.")

    else:

        product = product.strip().lower()

        # Find rules where entered product is in antecedent
        result = rules[
            rules["antecedents"].apply(
                lambda x: product in [i.lower() for i in x]
            )
        ]

        if result.empty:
            st.error("No recommendation found.")

        else:

            # Select highest confidence rule
            best = result.sort_values(
                by=["confidence", "lift"],
                ascending=False
            ).iloc[0]

            recommended = ", ".join(best["consequents"])

            st.success(
                f"Customers who bought **{product.title()}** most frequently also bought:"
            )

            st.metric(
                label="Recommended Product",
                value=recommended
            )

            st.write("### Statistics")
            st.write(f"**Confidence:** {best['confidence']:.2%}")
            st.write(f"**Support:** {best['support']:.2%}")
            st.write(f"**Lift:** {best['lift']:.2f}")
