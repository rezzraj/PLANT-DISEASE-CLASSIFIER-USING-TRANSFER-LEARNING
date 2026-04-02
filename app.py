import streamlit as st
from PIL import Image
from backend import predict

st.set_page_config(page_title="Plant Disease Detector")

st.title("🌿 Plant Disease Detector")

# Accepted classes
plant_diseases = {
    "Cherry": ["Powdery mildew", "Healthy"],
    "Corn (maize)": [
        "Cercospora leaf spot / Gray leaf spot",
        "Common rust",
        "Northern Leaf Blight",
        "Healthy"
    ],
    "Grape": [
        "Black rot",
        "Esca (Black Measles)",
        "Leaf blight (Isariopsis Leaf Spot)",
        "Healthy"
    ],
    "Orange": ["Haunglongbing (Citrus greening)"],
    "Peach": ["Bacterial spot", "Healthy"],
    "Pepper, bell": ["Bacterial spot", "Healthy"],
    "Potato": ["Early blight", "Late blight", "Healthy"],
    "Strawberry": ["Leaf scorch", "Healthy"],
    "Tomato": [
        "Bacterial spot",
        "Early blight",
        "Late blight",
        "Leaf Mold",
        "Septoria leaf spot",
        "Spider mites Two-spotted spider mite",
        "Target Spot",
        "Tomato Yellow Leaf Curl Virus",
        "Tomato mosaic virus",
        "Healthy"
    ],
}

with st.expander("🍃 Plants and diseases this model can predict"):
    for plant, diseases in plant_diseases.items():
        st.markdown(f"**{plant}**")
        for disease in diseases:
            st.write(f"- {disease}")

# upload option
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# camera option
camera_image = st.camera_input("Take Photo")

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


