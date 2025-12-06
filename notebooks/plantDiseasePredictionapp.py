import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model_tuned = tf.keras.models.load_model("..\\models\model_mobileNet_20Epochs_finetuned.h5")
splitDest = "E:\\MS_USD\\Course 8_AAI-521_ComputerVision\\Final Project\\AAI-521-Group2_Project\\DatasetForDiseasePrediction\\ImagesSplitIntoTrain_Test"
classNames = ['anthracnose',
 'bacterial blight',
 'brown spot',
 'green mite',
 'gumosis',
 'leaf blight',
 'leaf curl',
 'leaf miner',
 'leaf spot',
 'mosaic',
 'red rust',
 'septoria leaf spot',
 'streak virus',
 'verticulium wilt']

def predictDisease(imgPath, ind):
    # img_path = splitDest + imgPath
    img_path = imgPath
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)    
    pred = model_tuned.predict(img_array)
    classNameInd = np.argmax(pred[0])
    # print(classNameInd)
    confidence = round(np.max(pred[0]),4)
    print(classNames[classNameInd],confidence)
    # plt.subplot(5,3,ind)
    # plt.imshow(img)
    # plt.title(f"{classNames[classNameInd]}, Confidence Score: {confidence*100} %")
    return classNames[classNameInd],confidence

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
    print(uploaded_file)
    image = Image.open(uploaded_file)
    image = image.resize((224,224))
    st.image(image, caption="Uploaded Image", use_column_width=False)
    diseaseClass, conf = predictDisease(uploaded_file, 0)
    st.header(f"**Disease Identified** : {diseaseClass}")
    st.write(f"**Confidence score** : {conf*100} %")
    st.write(f"**CNN Model Used** : MobileNetV2")
    st.subheader("📊 Image Details")
    st.write(f"**Format:** {image.format}")
    st.write(f"**Size (pixels):** {image.size[0]} x {image.size[1]}")
    st.write(f"**Mode:** {image.mode}")

    image_array = np.array(image)
    st.write("Array shape:", image_array.shape)
else:
    st.info("⬆️ Upload an image to get started.")
