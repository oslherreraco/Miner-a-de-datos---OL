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

# Nombres de las columnas (según el dataset boston_housing)
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

# Crear una función que construya la interfaz y haga la predicción
def predict_price(model):
    # Título de la app
    st.title('Predicción del valor de una vivienda')
    
    # Explicación breve
    st.write("Introduce los datos de la vivienda para estimar su precio promedio.")
    
    # Crear el formulario de entrada en 4 columnas
    cols = st.columns(4)

    # Inicializar los inputs vacíos solo cuando se haga clic en "Iniciar"
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {col: 0.0 for col in columns}
        
        
        # Mostrar los datos ingresados en una tabla
        st.write("Valores introducidos en la tabla:")
        st.dataframe(pd.DataFrame(st.session_state.inputs, index=[0]))
        
        # Botón "Registrar y Predecir"
        if st.button("Registrar y Predecir"):
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
