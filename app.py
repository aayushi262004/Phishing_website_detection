import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib

from feature_extractor import extract_all_features

# ---------------------------------------------------------
# LOAD MODEL + SCALERS + FEATURE GROUPS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("three_branch_phishing_detector.h5")

@st.cache_resource
def load_scalers():
    url_scaler = joblib.load("url_scaler.pkl")
    dom_scaler = joblib.load("dom_scaler.pkl")
    network_scaler = joblib.load("network_scaler.pkl")
    return url_scaler, dom_scaler, network_scaler

@st.cache_resource
def load_feature_groups():
    return joblib.load("feature_groups.pkl")


model = load_model()
url_scaler, dom_scaler, network_scaler = load_scalers()
feature_groups = load_feature_groups()

# ---------------------------------------------------------
# FIX URL FEATURE LIST → EXACT 40 USED DURING TRAINING
# ---------------------------------------------------------
CORRECT_URL_FEATURES = [
    'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
    'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
    'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
    'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
    'qty_percent_url', 'qty_tld_url', 'length_url', 'qty_dot_domain',
    'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain',
    'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain',
    'qty_and_domain', 'qty_exclamation_domain', 'qty_space_domain',
    'qty_tilde_domain', 'qty_comma_domain', 'qty_plus_domain',
    'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
    'qty_percent_domain', 'qty_vowels_domain', 'server_client_domain',
    'domain_length', 'domain_in_ip'
]

feature_groups["url_features"] = CORRECT_URL_FEATURES  # override wrong pkl list


# ---------------------------------------------------------
# PREDICTION ENGINE
# ---------------------------------------------------------
def predict_url(url):

    # extract features from the URL
    df = extract_all_features(url)

    # build required columns for all 3 branches
    full_required = (
        feature_groups["url_features"]
        + feature_groups["dom_features"]
        + feature_groups["network_features"]
    )

    # create missing columns
    for col in full_required:
        if col not in df.columns:
            df[col] = 0

    # reorder exactly like model training
    df = df[full_required]

    # split branches
    url_data = df[feature_groups["url_features"]].values
    dom_data = df[feature_groups["dom_features"]].values
    net_data = df[feature_groups["network_features"]].values

    # DEBUG INFO (visible on Streamlit)
    st.write("### 🔍 DEBUG INFO (URL Features)")
    st.write("URL feature count:", len(feature_groups["url_features"]))
    st.write("url_data shape:", url_data.shape)
    st.write("URL feature names:", feature_groups["url_features"])

    # scale data
    url_scaled = url_scaler.transform(url_data)
    dom_scaled = dom_scaler.transform(dom_data)
    net_scaled = network_scaler.transform(net_data)

    # model prediction
    prob = model.predict([url_scaled, dom_scaled, net_scaled])[0][0]
    return prob


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Phishing Detector", layout="centered")

st.title("🔐 AI-Based Phishing URL Detector")
st.write("Enter any URL to check if it is **legitimate** or **phishing**.")

url = st.text_input("Enter URL:", placeholder="https://example.com/login")

if st.button("Analyze"):
    if not url.strip():
        st.error("Please enter a URL.")
    else:
        with st.spinner("Analyzing URL with AI model..."):
            probability = predict_url(url)

        st.success("Analysis Complete!")

        st.subheader("🧠 Prediction Result:")
        if probability > 0.50:
            st.error(f"🚨 High Risk: Phishing Detected!\nProbability: {probability:.2%}")
        else:
            st.success(f"✔ Safe: Likely Legitimate\nProbability: {probability:.2%}")

        st.write("---")
        st.caption("Made with ❤️ by Poppo, using a 3-branch multi-modal phishing detection model.")
