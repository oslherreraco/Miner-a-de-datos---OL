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
    if "step" not in st.session_state:
        st.session_state.step = 0  # Indicar el paso actual (empezar en 0)
    
    if "input_data" not in st.session_state:
        st.session_state.input_data = {col: None for col in columns}  # Inicializar todos los valores a None

    # Mostrar los campos que ya han sido diligenciados
    for i in range(st.session_state.step):
        col = columns[i]
        st.write(f"{col}: {st.session_state.input_data[col]}")  # Mostrar los valores ya ingresados

    # Mostrar el campo de entrada para la variable correspondiente al paso actual
    current_col = columns[st.session_state.step]
    
    # Obtener el valor previamente ingresado o mantener "" si es la primera vez
    previous_value = st.session_state.input_data[current_col] if st.session_state.input_data[current_col] is not None else ""
    
    # Crear el campo de entrada con el valor previo (si lo hay)
    input_value = st.text_input(f"Ingrese el valor para {current_col}", value=str(previous_value))

    # Validar que la entrada sea un número
    if input_value != "":
        try:
            st.session_state.input_data[current_col] = float(input_value)  # Guardar el valor ingresado
        except ValueError:
            st.warning(f"Por favor ingrese un valor numérico válido para {current_col}.")
    
    # Botón "Siguiente"
    if st.button("Siguiente"):
        # Avanzar al siguiente paso
        if st.session_state.step < len(columns) - 1:
            st.session_state.step += 1
        else:
            # Realizar la predicción al llegar al último paso
            input_array = np.array([list(st.session_state.input_data.values())])
            prediction = model.predict(input_array)
            st.write(f"El valor estimado de la vivienda es: ${prediction[0]:,.2f}")
            
            # Resetear el flujo para permitir nuevas predicciones
            st.session_state.step = 0
            st.session_state.input_data = {col: None for col in columns}

def main():
    # Cargar el modelo
    model = load_model()

    # Ejecutar la predicción
    predict_price(model)

# Si el script es ejecutado directamente, se llama a main()
if __name__ == "__main__":
    main()
