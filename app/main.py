import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(page_title="ByteGuard | Malware Vision", page_icon="🛡️", layout="wide")

# --- Load Custom CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("app/styling.css")
except:
    pass 

# --- Load Models & Classes ---
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cnn_path = os.path.join(base_dir, "src", "models", "byteguard_cnn.keras")
    rf_path = os.path.join(base_dir, "src", "models", "byteguard_rf.joblib")
    svm_path = os.path.join(base_dir, "src", "models", "byteguard_svm.joblib")
    
    cnn = tf.keras.models.load_model(cnn_path)
    rf = joblib.load(rf_path)
    svm = joblib.load(svm_path)
    return cnn, rf, svm

cnn_model, rf_model, svm_model = load_models()

CLASS_NAMES = [
    "Adialer.C", "Agent.FYI", "Allaple.A", "Allaple.L", "Alueron.gen!J", "Autorun.K", 
    "C2LOP.P", "C2LOP.gen!g", "Dialplatform.B", "Dontovo.A", "Fakerean", "Instantaccess", 
    "Lolyda.AA1", "Lolyda.AA2", "Lolyda.AA3", "Lolyda.AT", "Malex.gen!J", "Obfuscator.AD", 
    "Rbot!gen", "Skintrim.N", "Swizzor.gen!E", "Swizzor.gen!I", "VB.AT", "Wintrim.BX", "Yuner.A"
]

# --- UI Header ---
st.title("🛡️ ByteGuard Intelligence Pipeline")
st.markdown("**Automated Malware Image Classification & Threat Analysis**")
st.markdown("---")

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["🔍 Threat Classifier", "📊 Analytics Dashboard", "🤖 Model Comparison"])

# --- TAB 1: Classifier ---
with tab1:
    st.header("Upload Malware Sample (Image Representation)")
    st.markdown('<div class="left-align"><p>Upload a grayscale malware image (.png, .jpg)</p></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Malware Image", 
        type=["png", "jpg", "jpeg"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1.2, 2]) 
        
        with col1:
            image = Image.open(uploaded_file).convert('L')
            st.image(image, caption="Uploaded Binary Image", width="stretch")
            
        with col2:
            st.subheader("Intelligence Engine Execution")
            with st.spinner("Extracting spatial byte features..."):
                img_resized = image.resize((64, 64))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                
                cnn_input = img_array.reshape(1, 64, 64, 1)
                flat_input = img_array.reshape(1, 4096)
                
                cnn_pred = cnn_model.predict(cnn_input, verbose=0)
                cnn_class = CLASS_NAMES[np.argmax(cnn_pred)]
                cnn_conf = np.max(cnn_pred) * 100
                
                rf_pred_proba = rf_model.predict_proba(flat_input)
                rf_class = CLASS_NAMES[np.argmax(rf_pred_proba)]
                rf_conf = np.max(rf_pred_proba) * 100
                
                svm_class = CLASS_NAMES[svm_model.predict(flat_input)[0]]
                
            with st.container(border=True):
                st.markdown(f"**CNN Verdict:** {cnn_class}")
                st.progress(int(cnn_conf))
                st.caption(f"Confidence Score: {cnn_conf:.2f}%")
            
            with st.container(border=True):
                st.markdown(f"**Random Forest Verdict:** {rf_class}")
                st.progress(int(rf_conf))
                st.caption(f"Confidence Score: {rf_conf:.2f}%")
                
            with st.container(border=True):
                st.markdown(f"**Linear SVM Verdict:** {svm_class}")
                st.caption("Confidence Score: N/A (Geometric Hard Margin)")

# --- TAB 2: Analytics Dashboard ---
with tab2:
    st.header("Dataset Analytics & Engineering")
    st.markdown("Visualizing the processed subset to ensure balanced class distributions and valid structural byte representations.")
    
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    d_col1.metric("Total Records", "5,160", "Downsampled from 9,339")
    d_col2.metric("Threat Families", "25", "Categorized Classes")
    d_col3.metric("Train/Test Split", "80 / 20", "Stratified")
    d_col4.metric("Image Resolution", "64x64", "Grayscale Normalized")
    
    st.divider()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dist_path = os.path.join(base_dir, "docs", "assets", "class_distribution.png")
    samples_path = os.path.join(base_dir, "docs", "assets", "sample_images.png")
    
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.subheader("Class Imbalance Mitigation")
            st.markdown("Dominant families (e.g., Allaple.A) were capped at a maximum of 400 samples to prevent model bias and overfitting during training.")
            if os.path.exists(dist_path):
                st.image(dist_path, width="stretch")
    with col_b:
        with st.container(border=True):
            st.subheader("Spatial Feature Extraction")
            st.markdown("Raw malware binaries converted to 2D grayscale textures to expose distinct code structures, padding, and byte arrangements.")
            if os.path.exists(samples_path):
                st.image(samples_path, width="stretch")

# --- TAB 3: Model Comparison ---
with tab3:
    st.header("Comparative Performance Metrics")
    st.markdown("Evaluation on the 1,032-record isolated test set.")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Random Forest Accuracy", "97.09%", "+1.26% over CNN")
    m_col2.metric("CNN Accuracy", "95.83%", "+0.77% over SVM")
    m_col3.metric("Linear SVM Accuracy", "95.06%", "-2.03% under RF")
    
    st.divider()
    
    st.subheader("Detailed Classification Metrics")
    det_col1, det_col2, det_col3 = st.columns(3)
    
    with det_col1:
        st.markdown("**Random Forest (Best)**")
        st.markdown("- **Weighted F1-Score:** 97.00%")
        st.markdown("- **Macro Precision:** 96.00%")
        st.markdown("- **Macro Recall:** 96.00%")
        
    with det_col2:
        st.markdown("**Convolutional Neural Network**")
        st.markdown("- **Weighted F1-Score:** 95.00%")
        st.markdown("- **Macro Precision:** 92.00%")
        st.markdown("- **Macro Recall:** 93.00%")
        
    with det_col3:
        st.markdown("**Linear SVM (Baseline)**")
        st.markdown("- **Weighted F1-Score:** 95.00%")
        st.markdown("- **Macro Precision:** 93.00%")
        st.markdown("- **Macro Recall:** 93.00%")
        
    st.info("The Random Forest ensemble proved highly effective at mapping the flattened 4,096 byte-intensity features to specific malware families, outperforming the spatial feature extraction of the 3-layer Convolutional Neural Network.")
