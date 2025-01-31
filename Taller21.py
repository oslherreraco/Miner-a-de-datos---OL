import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import pandas as pd

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

    # Almacenamos la predicción y los hiperparámetros
    predicted_class = None
    model_params = None

    if uploaded_file is not None:
        # Abrir la imagen subida
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida")  # Mostrar la imagen subida

        # Preprocesar la imagen antes de clasificarla
        preprocessed_image = preprocess_image(image)  # La imagen ya tiene la forma correcta

        # Mostrar la imagen procesada (opcional)
        st.image(image, caption="Imagen preprocesada")  # Mostrar la imagen original (no tensor)

        if st.button("Clasificar imagen"):
            st.markdown("Imagen clasificada")
            model = load_model()  # Cargar el modelo

            if model is not None:
                # Aplanar la imagen a un vector de 784 características para modelos de scikit-learn
                flattened_image = preprocessed_image.reshape(1, -1)  # Convertir la imagen en un vector de 784 características

                # Realizar la predicción con el modelo cargado
                prediction = model.predict(flattened_image)  # La imagen ya tiene la forma correcta

                # Guardar la clase predicha
                predicted_class = prediction[0]  # Para modelos de clasificación
                st.markdown(f"La imagen fue clasificada como: {predicted_class}")

                # Si el modelo es de scikit-learn, puedes obtener los hiperparámetros
                if hasattr(model, 'get_params'):
                    model_params = model.get_params()

                    # Convertir los hiperparámetros a un formato adecuado para una tabla
                    model_params_table = [(key, value) for key, value in model_params.items()]

                    # Reemplazar <NA> o None por un guion o valor vacío
                    cleaned_model_params = [
                        (key, value if value is not None and value != "<NA>" else "-") 
                        for key, value in model_params_table
                    ]

                    # Convertir a un DataFrame de pandas para tener control sobre la tabla
                    df = pd.DataFrame(cleaned_model_params, columns=["Hiperparámetro", "Valor"])

                    # Mostrar un checkbox para que el usuario decida si quiere ver los hiperparámetros
                    show_params = st.checkbox("Mostrar hiperparámetros del modelo")

                    if show_params:
                        # Mostrar la tabla con los hiperparámetros cuando el checkbox esté marcado
                        st.write(df)

if __name__ == "__main__":
    main()
