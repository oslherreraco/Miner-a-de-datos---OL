import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle

# Diccionario para asignar nombres a las clases
class_names = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido",
    "Abrigo", "Sandalia", "Camisa", "Zapatilla deportiva",
    "Bolso", "Botín"
]

def preprocess_image(image):
    image = image.convert('L')  # convertir a escala de grises
    image = image.resize((28, 28))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def load_model():
    filename = "model_trained.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    # Título con color
    st.markdown(
        '<h1 style="color: #4CAF50; text-align: center;">Clasificación de la base de datos Fashion MNIST</h1>',
        unsafe_allow_html=True
    )
    st.markdown("Sube una imagen para clasificar")

    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown("### Imagen original:")
        st.image(image, caption="Imagen subida", use_container_width=True)

        # Preprocesar imagen
        preprocessed_image = preprocess_image(image)

        # Mostrar las imágenes lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen Original", use_container_width=True)
        with col2:
            st.image(
                preprocessed_image[0].reshape(28, 28),
                caption="Imagen Preprocesada",
                use_container_width=True
            )

        if st.button("Clasificar imagen"):
            model = load_model()
            prediction = model.predict(preprocessed_image.reshape(1, -1))  # (1, 784)
            class_id = np.argmax(prediction)  # Obtener índice de la clase predicha
            class_name = class_names[class_id]  # Obtener nombre de la clase

            st.markdown(f"### La imagen fue clasificada como: **{class_name}**")

if __name__ == "__main__":
    main()
