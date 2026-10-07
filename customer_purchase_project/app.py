import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Customer Purchase Prediction")

st.sidebar.markdown("""
### Machine Learning Project

**Model**
- XGBoost Classifier

**Features**
- 23 Engineered Features

**Modules**
- Prediction
- Product Recommendation
- Coupon Recommendation
- Customer Insights
""")

st.sidebar.success("Model Loaded Successfully ✅")

# ============================================================
# HEADER
# ============================================================

try:
    st.image("images/banner.png", use_container_width=True)
except:
    pass

st.title("🛒 Customer Purchase Prediction System")
st.caption("Predict customer purchasing behaviour using Machine Learning")

st.markdown("---")

# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.header("👤 Personal Information")

c1, c2 = st.columns(2)

with c1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with c2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

annual_income = st.number_input(
    "Annual Income",
    min_value=1000,
    value=50000,
    step=1000
)

# ============================================================
# PURCHASE DETAILS
# ============================================================

st.header("🛍 Purchase Details")

c1, c2, c3 = st.columns(3)

with c1:
    number_of_purchases = st.number_input(
        "Number of Purchases",
        min_value=0,
        value=10
    )

with c2:
    customer_tenure = st.number_input(
        "Customer Tenure (Years)",
        min_value=0,
        value=5
    )

with c3:
    last_purchase_days = st.number_input(
        "Days Since Last Purchase",
        min_value=0,
        value=30
    )

# ============================================================
# WEBSITE ACTIVITY
# ============================================================

st.header("🌐 Website Activity")

c1, c2 = st.columns(2)

with c1:
    time_on_website = st.number_input(
        "Time Spent on Website",
        min_value=0,
        value=25
    )

with c2:
    session_count = st.number_input(
        "Session Count",
        min_value=0,
        value=12
    )

# ============================================================
# CUSTOMER DETAILS
# ============================================================

st.header("⭐ Customer Details")

product_category = st.selectbox(
    "Product Category",
    [
        "Electronics",
        "Fashion",
        "Furniture",
        "Groceries",
        "Kitchen"
    ]
)

preferred_device = st.selectbox(
    "Preferred Device",
    [
        "Desktop",
        "Mobile",
        "Tablet"
    ]
)

region = st.selectbox(
    "Region",
    [
        "East",
        "North",
        "South",
        "West"
    ]
)

referral_source = st.selectbox(
    "Referral Source",
    [
        "Email",
        "Organic",
        "Paid Ads",
        "Referral",
        "Social"
    ]
)

customer_segment = st.selectbox(
    "Customer Segment",
    [
        "Premium",
        "Regular",
        "VIP"
    ]
)

loyalty_program = st.selectbox(
    "Loyalty Program",
    [
        "0",
        "1"
    ]
)

discounts_availed = st.number_input(
    "Discounts Availed",
    min_value=0,
    value=5
)

customer_satisfaction = st.slider(
    "Customer Satisfaction",
    1,
    5,
    4
)

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Customer Purchase",
    use_container_width=True
)
# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # -----------------------------
    # LABEL ENCODING
    # -----------------------------

    gender_value = 1 if gender == "Male" else 0
    loyalty_value = int(loyalty_program)

    product_mapping = {
        "Electronics": 0,
        "Fashion": 1,
        "Furniture": 2,
        "Groceries": 3,
        "Kitchen": 4
    }

    device_mapping = {
        "Desktop": 0,
        "Mobile": 1,
        "Tablet": 2
    }

    region_mapping = {
        "East": 0,
        "North": 1,
        "South": 2,
        "West": 3
    }

    referral_mapping = {
        "Email": 0,
        "Organic": 1,
        "Paid Ads": 2,
        "Referral": 3,
        "Social": 4
    }

    segment_mapping = {
        "Premium": 0,
        "Regular": 1,
        "VIP": 2
    }

    product_value = product_mapping[product_category]
    device_value = device_mapping[preferred_device]
    region_value = region_mapping[region]
    referral_value = referral_mapping[referral_source]
    segment_value = segment_mapping[customer_segment]

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------

    spending_index = annual_income * number_of_purchases
    engagement_score = time_on_website * session_count
    income_per_purchase = annual_income / (number_of_purchases + 1)
    loyalty_score = customer_tenure * number_of_purchases
    activity_score = session_count / (last_purchase_days + 1)
    discount_rate = discounts_availed / (number_of_purchases + 1)
    satisfied_loyal_customer = customer_satisfaction * customer_tenure

    # -----------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------

    input_data = pd.DataFrame({

        "Age":[age],
        "AnnualIncome":[annual_income],
        "NumberOfPurchases":[number_of_purchases],
        "TimeSpentOnWebsite":[time_on_website],
        "CustomerTenureYears":[customer_tenure],
        "LastPurchaseDaysAgo":[last_purchase_days],
        "Gender":[gender_value],
        "ProductCategory":[product_value],
        "PreferredDevice":[device_value],
        "Region":[region_value],
        "ReferralSource":[referral_value],
        "CustomerSegment":[segment_value],
        "LoyaltyProgram":[loyalty_value],
        "DiscountsAvailed":[discounts_availed],
        "SessionCount":[session_count],
        "CustomerSatisfaction":[customer_satisfaction],
        "SpendingIndex":[spending_index],
        "EngagementScore":[engagement_score],
        "IncomePerPurchase":[income_per_purchase],
        "LoyaltyScore":[loyalty_score],
        "ActivityScore":[activity_score],
        "DiscountRate":[discount_rate],
        "SatisfiedLoyalCustomer":[satisfied_loyal_customer]

    })

    # -----------------------------
    # CORRECT COLUMN ORDER
    # -----------------------------

    input_data = input_data[[
        "Age",
        "AnnualIncome",
        "NumberOfPurchases",
        "TimeSpentOnWebsite",
        "CustomerTenureYears",
        "LastPurchaseDaysAgo",
        "Gender",
        "ProductCategory",
        "PreferredDevice",
        "Region",
        "ReferralSource",
        "CustomerSegment",
        "LoyaltyProgram",
        "DiscountsAvailed",
        "SessionCount",
        "CustomerSatisfaction",
        "SpendingIndex",
        "EngagementScore",
        "IncomePerPurchase",
        "LoyaltyScore",
        "ActivityScore",
        "DiscountRate",
        "SatisfiedLoyalCustomer"
    ]]

    # -----------------------------
    # SCALE & PREDICT
    # -----------------------------

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = float(model.predict_proba(input_scaled)[0][1] * 100)
        # ============================================================
    # PREDICTION RESULT
    # ============================================================

    st.markdown("---")
    st.header("📊 Prediction Result")

    if prediction == 1:

        st.success("✅ Customer is Likely to Purchase")

        st.metric(
            "Purchase Probability",
            f"{probability:.2f}%"
        )

        st.progress(float(probability) / 100)

    else:

        st.error("❌ Customer is Unlikely to Purchase")

        st.metric(
            "Purchase Probability",
            f"{100-probability:.2f}%"
        )

        st.progress(float((100-probability) / 100))

    # ============================================================
    # CUSTOMER INSIGHTS
    # ============================================================

    st.markdown("---")
    st.header("📈 Customer Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Spending Index", f"{spending_index:,.0f}")

    with col2:
        st.metric("Engagement Score", f"{engagement_score:.2f}")

    with col3:
        st.metric("Loyalty Score", f"{loyalty_score:.2f}")

    # ============================================================
    # PRODUCT RECOMMENDATION
    # ============================================================

    st.markdown("---")
    st.header("🛍 Personalized Product Recommendations")

    recommendations = {

        "Electronics":[
            "🎧 Wireless Earbuds",
            "⌚ Smart Watch",
            "🔊 Bluetooth Speaker",
            "📱 Power Bank"
        ],

        "Fashion":[
            "👟 Sneakers",
            "⌚ Smart Watch",
            "👕 Premium T-Shirt",
            "🧥 Jacket"
        ],

        "Furniture":[
            "🪑 Office Chair",
            "📚 Bookshelf",
            "🛏 Study Table",
            "💡 Table Lamp"
        ],

        "Groceries":[
            "🥜 Healthy Snacks",
            "🥛 Organic Milk",
            "🍎 Fresh Fruits",
            "🥗 Vegetables"
        ],

        "Kitchen":[
            "🍳 Cookware Set",
            "🔪 Knife Set",
            "🥣 Storage Containers",
            "☕ Coffee Maker"
        ]
    }

    for item in recommendations[product_category]:
        st.write(item)

    # ============================================================
    # COUPON RECOMMENDATION ENGINE
    # ============================================================

    st.markdown("---")
    st.header("🎟 Coupon Recommendation Engine")

    if prediction == 1:

        if customer_segment == "VIP":

            coupon = "VIP30"
            discount = "30%"

        elif customer_segment == "Premium":

            coupon = "PREMIUM20"
            discount = "20%"

        else:

            coupon = "SAVE10"
            discount = "10%"

    else:

        if last_purchase_days > 60:

            coupon = "COME BACK40"
            discount = "40%"

        elif customer_satisfaction <= 2:

            coupon = "SORRY25"
            discount = "25%"

        else:

            coupon = "FIRST15"
            discount = "15%"

    st.success(f"🎉 Recommended Coupon : **{coupon}**")

    st.info(f"Discount Offered : **{discount} OFF**")

    # ============================================================
    # MARKETING STRATEGY
    # ============================================================

    st.markdown("---")
    st.header("📢 Suggested Marketing Strategy")

    if prediction == 1:

        st.success("High Value Customer")

        st.write("✅ Send Premium Membership Offer")
        st.write("✅ Cross-sell Similar Products")
        st.write("✅ Offer Cashback")
        st.write("✅ Loyalty Reward Points")

    else:

        st.warning("Customer Needs Engagement")

        st.write("📧 Personalized Email")
        st.write("📱 Push Notification")
        st.write("🔥 Flash Sale")
        st.write("🎁 Discount Coupon")
            # ============================================================
    # CUSTOMER SUMMARY
    # ============================================================

    st.markdown("---")
    st.header("👤 Customer Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "Annual Income",
            "Purchases",
            "Customer Tenure",
            "Last Purchase",
            "Product Category",
            "Preferred Device",
            "Customer Segment",
            "Purchase Probability"
        ],

        "Value": [
            age,
            gender,
            f"₹{annual_income:,}",
            number_of_purchases,
            customer_tenure,
            f"{last_purchase_days} days ago",
            product_category,
            preferred_device,
            customer_segment,
            f"{probability:.2f}%"
        ]
    })

    st.dataframe(summary, use_container_width=True)

    # ============================================================
    # DOWNLOAD REPORT
    # ============================================================

    st.markdown("---")
    st.header("📥 Download Prediction Report")

    report = pd.DataFrame({
        "Prediction Date": [datetime.now().strftime("%d-%m-%Y %H:%M")],
        "Prediction": [
            "Likely to Purchase"
            if prediction == 1
            else "Unlikely to Purchase"
        ],
        "Probability (%)": [round(probability, 2)],
        "Coupon": [coupon],
        "Discount": [discount],
        "Recommended Category": [product_category]
    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Report (CSV)",
        data=csv,
        file_name="customer_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ============================================================
    # INPUT DATA
    # ============================================================

    with st.expander("📋 View Model Input Data"):

        st.dataframe(input_data, use_container_width=True)

# ============================================================
# SIDEBAR INFO
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Project Statistics")

st.sidebar.metric("Total Features", "23")
st.sidebar.metric("Model", "XGBoost")
st.sidebar.metric("Feature Engineering", "7 Features")

st.sidebar.markdown("---")

st.sidebar.success("✔ Prediction Ready")

st.sidebar.info(
    """
This application predicts customer purchase behaviour using
Machine Learning and recommends suitable products,
marketing strategies and discount coupons.
"""
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


