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

# Nombres de las columnas (según el dataset boston_housing)
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

# Crear una función que construya la interfaz y haga la predicción
def predict_price(model):
    # Título de la app
    st.title('Predicción del valor de una vivienda')
    
    # Explicación breve
    st.write("Introduce los datos de la vivienda para estimar su precio promedio.")
    
    # Inicializar las variables en st.session_state si no están presentes
    for col in columns:
        if f"input_{col}" not in st.session_state:
            st.session_state[f"input_{col}"] = 0.0  # Inicializar cada variable individualmente con valor 0.0

    # Organizar la entrada en forma de tabla con 6 columnas
    input_data = {}

    cols = st.columns(6)  # Crear 6 columnas

    for i, col in enumerate(columns):
        # Determinar en qué columna se debe colocar la variable y su valor
        column_index = i % 3  # Alterna entre las 3 primeras columnas para nombres y las otras 3 para valores

        if column_index == 0:
            cols[column_index].write(f"**{col}**")  # Mostrar el nombre de la variable
        else:
            # Mostrar el campo de entrada solo en las columnas para los valores
            input_value = cols[column_index].text_input(f"Ingrese el valor para {col}", 
                                                       value=str(st.session_state[f'input_{col}']))  # Como texto para evitar botones

            # Convertir el valor ingresado a número (si es válido)
            try:
                input_value = float(input_value) if input_value else 0.0  # Usar 0.0 si no se ingresa valor
            except ValueError:
                input_value = 0.0  # En caso de que no se ingrese un número válido

            # Guardamos el valor en session_state
            st.session_state[f'input_{col}'] = input_value
            input_data[col] = input_value  # Guardar el valor en el diccionario de entrada

    # Botón "Predecir"
    if st.button("Predecir"):
        # Convertir los valores introducidos en una matriz numpy
        input_array = np.array([list(input_data.values())])
        
        # Realizamos la predicción
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
