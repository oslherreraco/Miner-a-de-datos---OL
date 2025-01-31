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
    
    # Crear un DataFrame vacío con las columnas necesarias
    data_input = pd.DataFrame(columns=columns)
    
    # Usar st.number_input para capturar los valores de las variables
    for col in columns:
        data_input[col] = st.number_input(col, value=0.0, step=0.1)

    # Mostrar la tabla de entrada
    st.dataframe(data_input)

    # Comprobar si el DataFrame tiene al menos una fila con datos
    if data_input.isnull().values.any():
        st.warning("Por favor, completa todos los campos antes de continuar.")
        return
    
    # Asegurarse de que hay datos en el DataFrame antes de acceder a la fila
    if data_input.empty:
        st.warning("Por favor, ingresa los datos para realizar la predicción.")
        return

    # Convertir la tabla de entrada a un array numpy
    input_data = np.array([data_input.iloc[0].values])

    # Hacer la predicción si el usuario ha ingresado todos los valores
    if st.button("Predecir el valor de la vivienda"):
        prediction = model.predict(input_data)
        st.write(f"El valor estimado de la vivienda es: ${prediction[0]:,.2f}")

def main():
    # Cargar el modelo
    model = load_model()

    # Ejecutar la predicción
    predict_price(model)

# Si el script es ejecutado directamente, se llama a main()
if __name__ == "__main__":
    main()
