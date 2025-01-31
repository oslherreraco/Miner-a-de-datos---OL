import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import sklearn

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28 píxeles
    image_array = img_to_array(image) / 255.0  # Normalizar los valores de píxeles
    image_array = np.expand_dims(image_array, axis=0)  # Añadir la dimensión de batch (1,)
    return image_array

# Función para cargar el modelo
def load_model():
    filename = "model_trained_classifier2.pkl.gz"  # Asegúrate de tener el modelo comprimido en .gz
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Función principal de la aplicación Streamlit
def main():
    st.title("Clasificación de la base de datos MNIST")
    st.markdown("Sube una imagen para clasificar")

    # Subir archivo de imagen
    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

    # Cargar el modelo
    model = load_model()

    # Variable para almacenar si el modelo ya fue clasificado
    classified = False

    # Si se ha subido una imagen
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida")  # Mostrar la imagen subida
        preprocessed_image = preprocess_image(image)  # Preprocesar la imagen

        # Mostrar la imagen preprocesada
        st.image(image, caption="Imagen preprocesada")  

        # Clasificar la imagen cuando se presiona el botón
        if st.button("Clasificar imagen"):
            st.markdown("Imagen clasificada")
            flattened_image = preprocessed_image.reshape(1, -1)  # Convertir la imagen en un vector de 784 características
            prediction = model.predict(flattened_image)  # Realizar la predicción
            predicted_class = prediction[0]  # Resultado de la clasificación
            st.markdown(f"La imagen fue clasificada como: {predicted_class}")
            classified = True  # Marcar como clasificado

        # Mostrar el checkbox para los hiperparámetros SOLO después de clasificar la imagen
        if classified:
            show_hyperparameters = st.checkbox("Mostrar Hiperparámetros del modelo")

            # Si el checkbox está marcado, mostrar los hiperparámetros
            if show_hyperparameters:
                st.subheader("Hiperparámetros del Modelo:")
                model_params = model.get_params()  # Obtener los hiperparámetros

                # Limpiar los valores "<NA>" y "None" para mostrarlos como "-"
                cleaned_model_params = [
                    (key, value if value is not None and value != "<NA>" else "-") 
                    for key, value in model_params.items()
                ]
                
                # Mostrar la tabla con los hiperparámetros
                st.table(cleaned_model_params)

if __name__ == "__main__":
    main()
