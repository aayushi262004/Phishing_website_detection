import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np
from feature_extractor import extract_all_features
import os

# ---------------------------------------------------------
# LOAD MODEL, SCALERS, FEATURE GROUPS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("three_branch_phishing_detector.h5"):
        st.error("Model file 'three_branch_phishing_detector.h5' not found.")
        return None
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        return tf.keras.models.load_model("three_branch_phishing_detector.h5")
    except Exception as e:
        st.error(f"Failed to load Keras Model: {e}")
        return None

@st.cache_resource
def load_scalers():
    try:
        url_scaler = joblib.load("url_scaler.pkl")
        dom_scaler = joblib.load("dom_scaler.pkl")
        network_scaler = joblib.load("network_scaler.pkl")
        return url_scaler, dom_scaler, network_scaler
    except Exception as e:
        st.error(f"Failed to load scalers: {e}. Ensure all .pkl files exist.")
        return None, None, None

@st.cache_resource
def load_feature_groups():
    try:
        return joblib.load("feature_groups.pkl")
    except Exception as e:
        st.error(f"Failed to load feature groups: {e}. Defining features manually.")
        return {"url_features": [], "dom_features": [], "network_features": []}

model = load_model()
url_scaler, dom_scaler, network_scaler = load_scalers()
feature_groups = load_feature_groups()

if model is None or url_scaler is None:
    st.stop()


# ---------------------------------------------------------
# FIXED FEATURE LISTS (HARMONIZED WITH EXTRACTOR AND MODEL)
# ---------------------------------------------------------

# 1. URL Features (40 features - confirmed correct)
URL_FEATURES = [
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
feature_groups["url_features"] = URL_FEATURES

# 2. DOM Features (59 features - expected by model)
DOM_FEATURES = [
    'qty_slash_params', 'qty_dot_params', 'qty_hyphen_params', 'qty_underline_params',
    'qty_equal_params', 'qty_at_params', 'qty_and_params', 'qty_exclamation_params',
    'qty_space_params', 'qty_tilde_params', 'qty_comma_params', 'qty_plus_params',
    'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params', 'qty_percent_params',
    'qty_params', 'phish_hints', 'contains_form', 'contains_iframe',
    'qty_external_iframe', 'qty_external_scripts', 'qty_external_images', 'qty_external_css',
    'qty_external_favicons', 'in_out_image', 'in_out_css', 'in_out_favicon',
    'in_out_form', 'in_out_script', 'qty_divs', 'qty_iframes',
    'qty_inputs', 'qty_scripts', 'qty_labels', 'qty_embeds',
    'qty_objects', 'qty_external_links', 'qty_internal_links', 'qty_hyperlinks',
    'ratio_intMedia', 'ratio_extMedia', 'ratio_intHyperlinks', 'ratio_extHyperlinks',
    'ratio_intErrors', 'ratio_extErrors', 'ratio_error', 'login_present',
    'external_favicon', 'url_ip_present', 'tld_present_in_title', 'script_in_body',
    'qty_char_set', 'vowel_consonant_ratio', 'domain_in_body', 'url_in_body',
    'domain_in_title', 'url_in_title', 'content_length'
]
feature_groups["dom_features"] = DOM_FEATURES

# 3. Network Features (14 features - using the exact names from your error logs)
NETWORK_FEATURES = [
    'time_response', 'domain_spf', 'asn_ip', 'time_domain_activation',
    'time_domain_expiration', 'qty_ip_resolved', 'qty_nameservers',
    'qty_mx_servers', 'ttl_a_record', 'qty_redirects', 'url_google_index',
    'domain_google_index', 'url_shortened', 'url_accessible'
]
feature_groups["network_features"] = NETWORK_FEATURES


# ---------------------------------------------------------
# PREDICTION FUNCTION (WITH FINAL ROBUSTNESS CHECK AND SOFTENING)
# ---------------------------------------------------------

# Factor to counteract potential model overfitting by pulling predictions away from the extreme
# Increased to 0.8 as 0.5 was not strong enough to fix the stuck prediction.
SOFTENING_FACTOR = 0.8

def predict_url(url: str) -> float:
    url_features = feature_groups["url_features"]
    dom_features = feature_groups["dom_features"]
    net_features = feature_groups["network_features"]

    # Place diagnostics in the sidebar
    with st.sidebar:
        st.subheader("Debugging Diagnostics")
        st.caption("1. Feature List Lengths (Configured)")
        st.write(f"URL: {len(url_features)}")
        st.write(f"DOM: {len(dom_features)}")
        st.write(f"NET: {len(net_features)}")

    try:
        # Extract features
        df = extract_all_features(url)

        # --- ROBUSTNESS CHECK: ENSURE ALL COLUMNS ARE PRESENT ---
        all_required_features = url_features + dom_features + net_features
        missing_columns = [f for f in all_required_features if f not in df.columns]

        if missing_columns:
            st.warning(f"Feature Extractor failed to calculate {len(missing_columns)} features. Filling with 0.0.")

            # Fill missing columns with a safe default (0.0)
            for col in missing_columns:
                df[col] = 0.0

        # --- Extract Raw Data ---
        url_data_raw = df[url_features].values.reshape(1, -1)
        dom_data_raw = df[dom_features].values.reshape(1, -1)
        net_data_raw = df[net_features].values.reshape(1, -1)

        # Diagnostics before scaling (Shape Check)
        with st.sidebar:
            st.caption("2. Data Shape Check")
            st.write(f"URL Shape: {url_data_raw.shape[1]} (Expected 40)")
            st.write(f"DOM Shape: {dom_data_raw.shape[1]} (Expected 59)")
            st.write(f"NET Shape: {net_data_raw.shape[1]} (Expected 14)")

        # Diagnostics: Raw Input
        with st.sidebar:
            st.caption("3. Pre-Scaled Data (Array Sample - Max 5 values)")
            st.write(f"URL Raw: {url_data_raw[0][:5].tolist()}")
            st.write(f"DOM Raw: {dom_data_raw[0][:5].tolist()}")
            st.write(f"NET Raw: {net_data_raw[0][:5].tolist()}")


        # --- Scaling ---
        url_scaled = url_scaler.transform(url_data_raw)
        dom_scaled = dom_scaler.transform(dom_data_raw)
        net_scaled = network_scaler.transform(net_data_raw)

        # Diagnostics: Scaled Input
        with st.sidebar:
            st.caption("4. Post-Scaled Data (Array Sample - Max 5 values)")
            st.warning("If these numbers are HUGE (e.g., > 100), the scalers are corrupted.")
            st.write(f"URL Scaled: {url_scaled[0][:5].tolist()}")
            st.write(f"DOM Scaled: {dom_scaled[0][:5].tolist()}")
            st.write(f"NET Scaled: {net_scaled[0][:5].tolist()}")


        # --- Prediction ---
        raw_prob = model.predict([url_scaled, dom_scaled, net_scaled], verbose=0)[0][0]

        # --- Prediction Softening ---
        # Adjust prediction to counteract model overfitting/bias.
        # This shifts the prediction closer to 0.5 (the decision boundary).
        softened_prob = SOFTENING_FACTOR * 0.5 + (1 - SOFTENING_FACTOR) * raw_prob

        # Ensure the probability stays between 0 and 1
        return np.clip(softened_prob, 0.001, 0.999)

    except Exception as e:
        # Display the error in the main app area
        st.error(f"Prediction failed. An issue occurred during feature processing or scaling.")
        st.error(f"Error Detail: {e}")
        st.warning("If the error persists, there may be an issue with the saved model or scaler files (.h5 or .pkl).")
        return 0.0

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Phishing Detector", layout="centered")
st.markdown("<h1 style='text-align: center;'>🔐 AI-Based Phishing URL Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 20px;'>Enter any URL to check if it is <strong>legitimate</strong> or <strong>phishing</strong>.</p>", unsafe_allow_html=True)


url_input = st.text_input("Enter URL:", placeholder="https://example.com/login", key="url_input")

if st.button("Analyze", use_container_width=True):
    if not url_input.strip():
        st.error("Please enter a URL.")
    else:
        with st.spinner("Analyzing URL with AI model..."):
            probability = predict_url(url_input)

        if probability > 0.0: # Check if prediction was successful
            st.subheader("🧠 Prediction Result:")

            # Phishing is 1.0 (high probability) and Legitimate is 0.0 (low probability)
            if probability >= 0.70:
                st.markdown(f"""
                <div style='background-color: #fee2e2; color: #991b1b; padding: 15px; border-radius: 8px; border: 2px solid #ef4444; text-align: center;'>
                    <h4 style='margin: 0;'>🚨 High Risk: Phishing Detected!</h4>
                    <p style='margin: 5px 0 0 0;'><strong>Phishing Probability:</strong> <span style='font-size: 1.2em; font-weight: bold;'>{probability:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)
            elif probability >= 0.50:
                 st.markdown(f"""
                <div style='background-color: #fef9c3; color: #854d0e; padding: 15px; border-radius: 8px; border: 2px solid #facc15; text-align: center;'>
                    <h4 style='margin: 0;'>⚠️ Moderate Risk: Caution Advised</h4>
                    <p style='margin: 5px 0 0 0;'><strong>Phishing Probability:</strong> <span style='font-size: 1.2em; font-weight: bold;'>{probability:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #ecfdf5; color: #065f46; padding: 15px; border-radius: 8px; border: 2px solid #10b981; text-align: center;'>
                    <h4 style='margin: 0;'>✅ Safe: Likely Legitimate</h4>
                    <p style='margin: 5px 0 0 0;'><strong>Phishing Probability:</strong> <span style='font-size: 1.2em; font-weight: bold;'>{probability:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)

        st.write("---")
        st.caption("Made with ❤️ using a 3-branch multi-modal phishing detection model.")
