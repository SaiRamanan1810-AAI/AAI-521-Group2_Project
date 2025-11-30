import streamlit as st

st.set_page_config(
    page_title="Plant Disease Prediction App",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🖼️ Plant Disease Prediction App")
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Upload an image and predict the disease.")

st.sidebar.header("📌 About")
st.sidebar.info(
    "This app lets you upload images (JPG, PNG) and view them instantly. "
    "It also shows metadata like format and size."
    "Finally it predicts the disease a particular plant has"
)

st.sidebar.markdown("**Developed by:** Team 2")
st.sidebar.markdown("**Powered by:** Streamlit + Computer Vision")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("📊 Image Details")
    st.write(f"**Format:** {image.format}")
    st.write(f"**Size (pixels):** {image.size[0]} x {image.size[1]}")
    st.write(f"**Mode:** {image.mode}")

    image_array = np.array(image)
    st.write("Array shape:", image_array.shape)
else:
    st.info("⬆️ Upload an image to get started.")
