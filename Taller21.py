import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle
import sklearn


def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28 píxeles
    image_array = img_to_array(image) / 255.0  # Normalizar los valores de píxeles
    image_array = np.expand_dims(image_array, axis=0)  # Añadir la dimensión de batch (1,)
    image_array = np.expand_dims(image_array, axis=-1)  # Añadir la dimensión del canal (1,)
    return image_array

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

def main():
  st.title("Clasificación de la base de datos MNIST")
  st.markdown("Sube una imagen para clasificar")

  uploaded_file = st.file_uploader("Selecciona una imagen 1 (PNG, JPG, JPEG:)", type = ["jpg", "png", "jpeg"])

  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "imagen subida")

    preprocessed_image = preprocess_image(image) # (1,28,28)
   
    st.image(preprocessed_image, caption = "imagen subida")

    if st.button("Clasificar imagen"):
      st.markdown("Imagen clasificada")
      model = load_model()
  
      if model is not None:
          # Realizar la predicción directamente con la imagen procesada
          prediction = model.predict(preprocessed_image)  # El preprocesado ya tiene la forma correcta
          st.markdown(f"La imagen fue clasificada como: {prediction[0]}")


if __name__ == "__main__":
  main()
