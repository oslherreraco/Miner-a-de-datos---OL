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
    
    # Crear el formulario de entrada con 3 columnas
    cols = st.columns(3)
    
    # Inicializar el diccionario para capturar los valores de entrada
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {col: 0.0 for col in columns}

    # Asignamos los inputs a las columnas en el orden de izquierda a derecha
    with cols[0]:
        st.session_state.inputs["CRIM"] = st.number_input("CRIM", value=st.session_state.inputs["CRIM"], step=0.1)
        st.session_state.inputs["ZN"] = st.number_input("ZN", value=st.session_state.inputs["ZN"], step=0.1)
        st.session_state.inputs["INDUS"] = st.number_input("INDUS", value=st.session_state.inputs["INDUS"], step=0.1)
        st.session_state.inputs["CHAS"] = st.number_input("CHAS", value=st.session_state.inputs["CHAS"], step=0.1)
        st.session_state.inputs["NOX"] = st.number_input("NOX", value=st.session_state.inputs["NOX"], step=0.1)

    with cols[1]:
        st.session_state.inputs["RM"] = st.number_input("RM", value=st.session_state.inputs["RM"], step=0.1)
        st.session_state.inputs["AGE"] = st.number_input("AGE", value=st.session_state.inputs["AGE"], step=0.1)
        st.session_state.inputs["DIS"] = st.number_input("DIS", value=st.session_state.inputs["DIS"], step=0.1)
        st.session_state.inputs["RAD"] = st.number_input("RAD", value=st.session_state.inputs["RAD"], step=0.1)
        st.session_state.inputs["TAX"] = st.number_input("TAX", value=st.session_state.inputs["TAX"], step=0.1)

    with cols[2]:
        st.session_state.inputs["PTRATIO"] = st.number_input("PTRATIO", value=st.session_state.inputs["PTRATIO"], step=0.1)
        st.session_state.inputs["B"] = st.number_input("B", value=st.session_state.inputs["B"], step=0.1)
        st.session_state.inputs["LSTAT"] = st.number_input("LSTAT", value=st.session_state.inputs["LSTAT"], step=0.1)
    
    # Botón para limpiar los datos
    if st.button("Limpiar los datos"):
        # Limpiar todos los valores de entrada (ponerlos en 0.0)
        st.session_state.inputs = {col: 0.0 for col in columns}

    # Mostrar la tabla de entrada después de que el usuario ingrese todos los valores
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
