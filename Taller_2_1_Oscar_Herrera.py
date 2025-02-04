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
    filename = "model_trained_classifier.pkl.gz"  # Asegúrate de tener el modelo comprimido en .gz
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Función principal de la aplicación Streamlit
def main():
    st.title("Clasificación de la base de datos MNIST")

    if st.checkbox("## Ver análisis del modelo"):
        st.write("#### Análisis de hiperparámetros")
        st.markdown("""La búsqueda del mejor modelo para clasificación de imágenes de números manuscritos, particularmente empleando la base de datos MNIST, se realizó probando los métodos de vecinos más cercanos y árboles de decisión, en ambos casos tanto para variables en su dimensión original, esto es, sin escalar, como también aplicando técnicas de escalado como normalización (StandarScaler) o redimensionando en función de mínimos y máximos (MinMaxScaler)

Para el primer caso, vecinos más cercanos, el algoritmo inicialmente propuesto para el presente taller proponía opciones de número de vecinos pares, lo que puede dar lugar a aleatoriedad dado que este método clasifica por “votación” lo que da la posibilidad de empate. Ante esta circunstancia, se incluyeron también valores de cantidades impares de vecinos más cercanos entre 3 y 9 (2,3,4,5,6,7,8,9,10,20,30,40,50,100). Asimismo, se propusieron parámetros para emplear diferentes tipos de distancia para la medida de la cercanía de cada punto, considerando en este caso, la distancia de Manhattan, la distancia Euclidiana y la distancia de Minkowski, según tome p diferentes valores (1,2,3,4,5).

En el caso del método de árboles de decisión, se propuso la búsqueda del mejor modelo con base en la profundidad o “ramificación” de éste, tomando diferentes valores entre 1 y 100 (1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100).

Con esta información el mejor modelo obtenido fue evaluado con la precisión de predicción medida por su accuracy, obteniéndose un 0,8717 como la mejor opción para ser implementado en esta aplicación, lo cual indica que aproximadamente un 87,17% de las imágenes son clasificadas correctamente, según la contrastación realizada en la partición de prueba, comportamiento que se prevé similar para esta aplicación. 

El mejor modelo encontrado corresponde al método de vecinos más cercanos (KNeighborsClassifier) que indica que las distancias medidas en relación a los diferentes puntos de las nubes respectivas son mejores para identificar las imágenes en este caso que la división sucesiva realizada por el método de árboles de decisión; en otras palabras, se obtiene una clasificación más acertada cuando se compara con los puntos o registros existentes que se encuentran más próximos, lo que da una idea de la diferenciación de las nubes de puntos, más aún cuando este mejor modelo no requirió escalamiento de las variables. Esta última condición puede estar determinada por dimensiones similares de las variables de la base de datos o ser indicativa de que la escala de las mismas permite una adecuada separación de las clases.

Por su parte, el mejor hiperparámetro que se obtuvo en cuanto a cantidad de vecinos a considerar fue de 4. Pese a ser un número par, debe decirse que se probó en igualdad de condiciones con cantidades impares, y esto indica la cantidad de puntos o registros existentes contra los cuales se debe medir la distancia de un nuevo punto, en este caso una nueva imagen, para poder obtener una clasificación adecuada en cuanto al número o dígito que representa.

Ahora bien, la distancia a medir que permite una mejor clasificación es la distancia de Minkowski comparada con las distancias de Manhattan y la Euclidiana que también fueron probadas. La distancia de Minkowski es en realidad una generalización de distancias y coincide en p=1 con la de Manhattan y en p=2 con la Euclidiana mientras que para p>2 se utilizaría la referenciada, que es especialmente útil por cuanto es más sensible a diferencias grandes y menos sensible a diferencias pequeñas, lo que en el caso de la clasificación de imágenes de números manuscritos como el que se propone puede permitir “alejar” los grupos o clases para una identificación más clara.

Por último, y aunque no se probó en la búsqueda del modelo, se anota que los pesos de las distancias fueron establecidos en el valor por defecto, uniforme, es decir, que no se generó una diferenciación de algunos de los puntos o registros a comparar, lo cual es consistente con la naturaleza de la base de datos en los cuales se busca clasificar imágenes en igualdad de condiciones.
""")

    
    # Explicación breve
    st.write("#### Imagen")
   
    st.markdown("Sube una imagen para clasificar")

    # Subir archivo de imagen
    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

    if 'predicted_class' not in st.session_state:
        st.session_state.predicted_class = None  # Inicializamos el estado de la predicción

    if 'model_params' not in st.session_state:
        st.session_state.model_params = None  # Inicializamos el estado de los hiperparámetros

    # Mostrar la imagen subida y hacer predicción si la imagen es cargada
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
                st.session_state.predicted_class = prediction[0]  # Para modelos de clasificación
                st.markdown(f"#### La imagen fue clasificada como: {st.session_state.predicted_class}")

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
                    st.session_state.model_params = pd.DataFrame(cleaned_model_params, columns=["Hiperparámetro", "Valor"])

    # Checkbox para mostrar los hiperparámetros
    if st.checkbox("Mostrar hiperparámetros del modelo"):
        st.write("#### Hiperparámetros del modelo")
        if st.session_state.model_params is not None:
            # Estilo HTML para controlar el ancho de las columnas
            st.markdown(
                """
                <style>
                .dataframe th, .dataframe td {
                    padding: 10px;
                    text-align: left;
                    width: 300px;
                }
                </style>
                """, unsafe_allow_html=True
            )

            # Mostrar la tabla con estilo CSS para un ancho adecuado
            st.write(st.session_state.model_params.to_html(index=False, escape=False), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
