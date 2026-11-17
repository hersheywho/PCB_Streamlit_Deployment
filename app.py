import streamlit as st
import numpy as np
import cv2
import pickle
import joblib
from skimage.feature import hog
from PIL import Image

st.set_page_config(page_title="PCB Defect Classifier", page_icon="🔍", layout="centered")

@st.cache_resource
def load_models():
    svm = joblib.load("pcb_svm_model.pkl")
    tree = joblib.load("pcb_decision_tree_model.pkl")
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return svm, tree, class_names

def extract_features(image):
    # Keep preprocessing identical to the Colab training code:
    # resize to 64x64, grayscale, then HOG.
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    return features.reshape(1, -1)

st.title("🔍 PCB Defect Classification")
st.write("Upload a cropped PCB defect image. The models will classify it into one of the six trained defect categories.")

st.info("Important: This is a defect-crop classifier. Upload a close-up crop containing one suspected defect, not an entire PCB board.")

try:
    svm_model, tree_model, class_names = load_models()
except Exception as e:
    st.error("Could not load the model files. Check that all three .pkl files are in the same GitHub folder as app.py.")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader("Upload a PCB defect crop", type=["png", "jpg", "jpeg", "bmp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict defect", type="primary"):
        x = extract_features(image)
        svm_prediction = int(svm_model.predict(x)[0])
        tree_prediction = int(tree_model.predict(x)[0])

        svm_label = class_names.get(svm_prediction, str(svm_prediction))
        tree_label = class_names.get(tree_prediction, str(tree_prediction))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SVM + PCA")
            st.success(svm_label)
        with col2:
            st.subheader("Decision Tree")
            st.warning(tree_label)

        st.caption("The SVM + PCA model achieved about 73.5% accuracy and the Decision Tree about 45.4% on the test split reported by the training run. These are test-set results, not a guarantee for every uploaded image.")
else:
    st.caption("Choose an image above to start.")
