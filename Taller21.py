import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import pandas as pd  # Usamos pandas para crear tablas personalizadas

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

                # Mostrar el resultado de la predicción
                predicted_class = prediction[0]  # Para modelos de clasificación
                st.markdown(f"La imagen fue clasificada como: {predicted_class}")

                # Agregar un checkbox para mostrar u ocultar los hiperparámetros
                if st.checkbox("Mostrar hiperparámetros del modelo", value=False):

                    st.subheader("Hiperparámetros del Modelo:")
    
                    # Obtener los hiperparámetros del modelo
                    model_params = model.get_params()
    
                            # Convertir los hiperparámetros a una lista de tuplas (nombre, valor)
                    model_params_table = [(key, value) for key, value in model_params.items()]
    
                            # Convertir la lista de tuplas en un DataFrame de pandas
                    df_params = pd.DataFrame(model_params_table, columns=["Hiperparámetro", "Valor"])
    
                            # Mostrar la tabla sin índices
                    st.table(df_params)  # Mostrar la tabla sin número de fila

if __name__ == "__main__":
    main()
