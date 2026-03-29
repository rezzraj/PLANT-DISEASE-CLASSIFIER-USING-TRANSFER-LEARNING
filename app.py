import streamlit as st
from PIL import Image
from backend import predict

st.set_page_config(page_title="Plant Disease Detector ")

st.title(" Plant Disease Detector")

# upload option
uploaded_file = st.file_uploader("Upload Image ", type=["jpg", "png", "jpeg"])

# camera option
camera_image = st.camera_input("Take Photo ")

image = None

if uploaded_file:
    image = Image.open(uploaded_file)

elif camera_image:
    image = Image.open(camera_image)

if image:
    st.image(image, caption="Selected Image", use_container_width=True)

    if st.button("Predict 🔥"):
        with st.spinner("Analyzing... 🧠"):
            label, confidence = predict(image)
            label = label.replace("___", " - ").replace("_", " ")
            if confidence < 0.6:
                st.warning("⚠️ Low confidence. Try a clearer image.")

        st.success(f"🌿 Prediction: {label}")
        st.info(f"Confidence: {confidence * 100:.2f}%")


