import streamlit as st
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Product Recommendation",
    page_icon="🛒",
    layout="centered"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🛒 Product Recommendation System")

st.write(
    "Select a product to get a recommendation."
)


# ==========================================================
# TRANSACTION DATA
# ==========================================================

transactions = [
    ["Bread", "Milk"],
    ["Bread", "Diaper", "Beer", "Eggs"],
    ["Milk", "Diaper", "Beer", "Cola"],
    ["Bread", "Milk", "Diaper", "Beer"],
    ["Bread", "Milk", "Diaper", "Cola"]
]


# ==========================================================
# ONE-HOT ENCODING
# ==========================================================

te = TransactionEncoder()

te_array = te.fit(
    transactions
).transform(
    transactions
)

df = pd.DataFrame(
    te_array,
    columns=te.columns_
)


# ==========================================================
# APRIORI ALGORITHM
# ==========================================================

frequent_itemsets = apriori(
    df,
    min_support=0.2,
    use_colnames=True
)


# ==========================================================
# ASSOCIATION RULES
# ==========================================================

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)


# ==========================================================
# PRODUCT LIST
# ==========================================================

products = sorted(
    te.columns_
)


# ==========================================================
# USER INPUT
# ==========================================================

st.subheader("🛍️ Select Product")

product = st.selectbox(
    "Product Name",
    ["Select Product"] + products
)


# ==========================================================
# SEARCH BUTTON
# ==========================================================

if st.button(
    "🔍 Search",
    use_container_width=True
):

    # ======================================================
    # CHECK PRODUCT
    # ======================================================

    if product == "Select Product":

        st.warning(
            "⚠️ Please select a product."
        )

    else:

        # ==================================================
        # CONVERT PRODUCT TO LOWERCASE
        # ==================================================

        product_lower = product.lower()


        # ==================================================
        # FIND RULES
        # ==================================================

        result = rules[
            rules["antecedents"].apply(
                lambda x:
                product_lower in [
                    str(i).lower()
                    for i in x
                ]
            )
        ]


        # ==================================================
        # RESULT
        # ==================================================

        if result.empty:

            st.error(
                "❌ No recommendation found."
            )

        else:

            # ==============================================
            # SORT BY CONFIDENCE AND LIFT
            # ==============================================

            result = result.sort_values(
                by=["confidence", "lift"],
                ascending=False
            )


            # ==============================================
            # BEST RULE
            # ==============================================

            best = result.iloc[0]


            # ==============================================
            # RECOMMENDED PRODUCT
            # ==============================================

            recommended = ", ".join(
                sorted(
                    str(item)
                    for item in best["consequents"]
                )
            )


            # ==============================================
            # DISPLAY RESULT
            # ==============================================

            st.success(
                f"Customers who bought **{product}** also bought:"
            )


            st.metric(
                label="🛒 Recommended Product",
                value=recommended
            )


            # ==============================================
            # STATISTICS
            # ==============================================

            st.subheader("📊 Recommendation Details")

            st.write(
                f"**Confidence:** "
                f"{best['confidence']:.2%}"
            )

            st.write(
                f"**Support:** "
                f"{best['support']:.2%}"
            )

            st.write(
                f"**Lift:** "
                f"{best['lift']:.2f}"
            )
