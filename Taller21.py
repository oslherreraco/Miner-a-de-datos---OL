import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.convert('L')  # Convertir la imagen a escala de grises
    image = image.resize((28, 28))  # Redimensionar la imagen a 28x28 píxeles
    image_array = img_to_array(image) / 255.0  # Normalizar los valores de píxeles
    image_array = np.expand_dims(image_array, axis=0)  # Añadir la dimensión de batch (1,)
    image_array = np.expand_dims(image_array, axis=-1)  # Añadir la dimensión del canal (1,)
    return image_array

# Función para cargar el modelo
def load_model():
    filename = "model_trained_classifier2.pkl"
    try:
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Error: El archivo {filename} no se encuentra en el directorio.")
        return None
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Función principal
def main():
    st.title("Clasificación de la base de datos MNIST")
    st.markdown("Sube una imagen para clasificar")

    # Subir archivo de imagen
    uploaded_file = st.file_uploader("Selecciona una imagen 2 (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Abrir la imagen subida
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida")  # Mostrar la imagen subida

        # Preprocesar la imagen antes de clasificarla
        preprocessed_image = preprocess_image(image)

        # Mostrar la imagen procesada (opcional)
        st.image(image, caption="Imagen preprocesada")  # Mostrar la imagen original (no tensor)

        if st.button("Clasificar imagen"):
            st.markdown("Imagen clasificada")
            model = load_model()  # Cargar el modelo

            if model is not None:
                # Realizar la predicción con el modelo cargado
                prediction = model.predict(preprocessed_image)  # La imagen ya tiene la forma correcta
                st.markdown(f"La imagen fue clasificada como: {np.argmax(prediction)}")

# Ejecutar la aplicación de Streamlit
if __name__ == "__main__":
    main()

