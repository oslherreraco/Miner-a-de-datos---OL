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

# Calcular los promedios de las columnas del dataset de entrenamiento para imputar
column_means = train_data.mean(axis=0)

# Crear una función que construya la interfaz y haga la predicción
def predict_price(model):
    # Título de la app
    st.title('Predicción del valor de una vivienda')
    
    # Explicación breve
    st.write("Introduce los datos de la vivienda para estimar su precio promedio.")
    
    # Inicializar el diccionario para capturar los valores de entrada
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {col: 0.0 for col in columns}

    # Crear un dataframe para la tabla de entrada
    data = []
    for i in range(0, len(columns), 3):  # Esto organiza en filas de 3 columnas
        row = columns[i:i+3]
        data.append(row)

    # Crear una tabla con 3 columnas
    df = pd.DataFrame(data, columns=["Variable 1", "Variable 2", "Variable 3"])

    # Mostrar la tabla de entrada y capturar los valores
    st.write("Introduzca los datos para las variables de la vivienda:")
    st.dataframe(df)
    
    for idx, row in enumerate(data):
        for j, col in enumerate(row):
            value = st.number_input(f'{col} (Variable)', key=f'{col}_{idx}_{j}', value=st.session_state.inputs.get(col, 0.0), step=0.1)
            st.session_state.inputs[col] = value  # Guardar el valor ingresado

    # Botón para limpiar los datos
    if st.button("Limpiar los datos"):
        # Limpiar todos los valores de entrada (ponerlos en 0.0)
        st.session_state.inputs = {col: 0.0 for col in columns}

    # Mostrar los datos ingresados
    if st.button("Registrar y Predecir"):
        # Imputar valores faltantes con los promedios de las variables
        for col in st.session_state.inputs:
            if st.session_state.inputs[col] == 0.0:
                st.session_state.inputs[col] = column_means[columns.index(col)]
        
        # Mostrar los datos ingresados o imputados
        st.write("Valores introducidos en la tabla:")
        st.dataframe(pd.DataFrame(st.session_state.inputs, index=[0]))

        # Convertir los valores introducidos en una matriz numpy
        input_data = np.array([list(st.session_state.inputs.values())])
        
        # Realizamos la predicción
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
