import streamlit as st
import numpy as np
import pickle
import gzip
import pandas as pd

# Cargar el modelo entrenado
def load_model():
    with gzip.open('model_trained_regressor.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

# Nombres de las columnas y tipos de las variables (según el dataset boston_housing)
columns_types = {
    "CRIM": float,
    "ZN": float,
    "INDUS": float,
    "CHAS": bool,  # Variable booleana (1 o 0)
    "NOX": float,
    "RM": float,
    "AGE": float,
    "DIS": float,
    "RAD": float,
    "TAX": float,
    "PTRATIO": float,
    "B": float,
    "LSTAT": float
}

columns = list(columns_types.keys())

# Crear una función que construya la interfaz y haga la predicción
def predict_price(model):
    # Título de la app
    st.title('Predicción del valor de una vivienda')
    
    # Explicación breve
    st.write("Introduce el valor de una variable para estimar el precio de la vivienda.")
    
    # Inicializar las variables en st.session_state si no están presentes
    if "input_values" not in st.session_state:
        st.session_state.input_values = {col: None for col in columns}  # Inicializamos con None para cada variable

    # Crear el campo de entrada para cada variable
    input_data = {}
    
    for col in columns:
        col_type = columns_types[col]
        
        if col_type == float:
            # Para variables numéricas, usamos number_input
            input_data[col] = st.number_input(f"Ingrese el valor para {col}", value=st.session_state.input_values.get(col, 0.0))
        elif col_type == bool:
            # Para variables booleanas, usamos checkbox
            input_data[col] = st.checkbox(f"¿Está presente {col}?", value=st.session_state.input_values.get(col, False))
        
        # Actualizar session_state con el valor ingresado
        st.session_state.input_values[col] = input_data[col]
    
    # Botón "Predecir"
    if st.button("Predecir"):
        # Asegurarse de que todos los valores sean numéricos y convertir a un array de numpy
        input_array = np.array([list(input_data.values())])
        
        # Realizar la predicción
        prediction = model.predict(input_array)
        st.write(f"El valor estimado de la vivienda es: ${prediction[0]:,.2f}")

def main():
    # Cargar el modelo
    model = load_model()

    # Ejecutar la predicción
    predict_price(model)

# Si el script es ejecutado directamente, se llama a main()
if __name__ == "__main__":
    main()
