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

    # Estado de sesión para guardar la predicción y los hiperparámetros
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
        st.session_state.model_params = None
        st.session_state.show_checkbox = False  # Inicializamos el estado del checkbox

    if uploaded_file is not None:
        # Abrir la imagen subida
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida")  # Mostrar la imagen subida

        # Preprocesar la imagen antes de clasificarla
        preprocessed_image = preprocess_image(image)  # La imagen ya tiene la forma correcta

        # Mostrar la imagen procesada (opcional)
        st.image(image, caption="Imagen preprocesada")  # Mostrar la imagen original (no tensor)

        # Botón para clasificar la imagen
        if st.button("Clasificar imagen"):
            st.markdown("Imagen clasificada")
            model = load_model()  # Cargar el modelo

            if model is not None:
                # Aplanar la imagen a un vector de 784 características para modelos de scikit-learn
                flattened_image = preprocessed_image.reshape(1, -1)  # Convertir la imagen en un vector de 784 características

                # Realizar la predicción con el modelo cargado
                prediction = model.predict(flattened_image)  # La imagen ya tiene la forma correcta

                # Guardar la predicción y los hiperparámetros en el estado de sesión
                st.session_state.prediction = prediction[0]  # Guardar la clase predicha

                # Guardar los hiperparámetros del modelo
                if hasattr(model, 'get_params'):
                    st.session_state.model_params = model.get_params()

                # Mostrar la predicción
                st.markdown(f"La imagen fue clasificada como: {st.session_state.prediction}")

                # Activar el checkbox después de clasificar la imagen
                st.session_state.show_checkbox = True  # Mostrar el checkbox después de clasificar la imagen

        # Mostrar el checkbox después de clasificar la imagen
        if st.session_state.show_checkbox:
            if st.checkbox("Mostrar hiperparámetros del modelo", value=False):
                if st.session_state.model_params:
                    st.subheader("Hiperparámetros del Modelo:")
                    model_params_table = [(key, value) for key, value in st.session_state.model_params.items()]
                    
                    # Mostrar la tabla con los hiperparámetros
                    st.table(model_params_table)  # Mostrar los hiperparámetros en formato de tabla

if __name__ == "__main__":
    main()
