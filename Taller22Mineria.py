import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gzip

# Cargar el modelo entrenado
def load_model():
    with gzip.open('model_trained_regressor.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar los datos de Boston
from tensorflow.keras.datasets import boston_housing

# Cargar los datos de entrenamiento para asegurarse de que la entrada del usuario sea coherente
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Nombres de las columnas (según el dataset boston_housing)
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

# Crear una función que construya la interfaz y haga la predicción
def predict_price(model):
    # Título de la app
    st.title('Predicción del valor de una vivienda')
    
    # Explicación breve
    st.write("Introduce los datos de la vivienda para estimar su precio promedio.")
    
    # Crear el formulario de entrada en 4 columnas: 2 para las variables y 2 para los valores ingresados
    data_input = pd.DataFrame(columns=columns)
    
    # Usar `st.columns` para crear columnas interactivas
    col1, col2 = st.columns(4)
    
    inputs = {}
    
    with col1:
        for i, col in enumerate(columns[0:7]):
            inputs[col] = st.number_input(f"{col} (Variable)", value=0.0, step=0.1)
    
    with col2:
        for i, col in enumerate(columns[7:]):
            inputs[col] = st.number_input(f"{col} (Variable)", value=0.0, step=0.1)
    
    # Mostrar la tabla de entrada
    st.write("Valores introducidos en la tabla:")
    st.dataframe(pd.DataFrame(inputs, index=[0]))
    
    # Convertir los valores introducidos en una matriz numpy
    input_data = np.array([list(inputs.values())])
    
    # Predicción cuando se presiona el botón
    if st.button("Predecir el valor de la vivienda"):
        # Realizamos la predicción
        prediction = model.predict(input_data)
        # Mostrar el valor predicho
        st.write(f"El valor estimado de la vivienda es: ${prediction[0]:,.2f}")

def main():
    # Cargar el modelo
    model = load_model()

    # Ejecutar la predicción
    predict_price(model)

# Si el script es ejecutado directamente, se llama a main()
if __name__ == "__main__":
    main()

