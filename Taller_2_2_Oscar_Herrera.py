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

# Tipos de variables (ahora incluye precisión sobre si son enteras, reales o categóricas ordinales)
column_types = {
    "CRIM": "Tasa de criminalidad por cada 100,000 habitantes (numérico real)",
    "ZN": "Proporción de terrenos residenciales zonificados para grandes parcelas (numérico real)",
    "INDUS": "Porcentaje de terrenos comerciales (numérico real)",
    "CHAS": "Variable binaria (0: No, 1: Sí) para la proximidad al río Charles (entero)",
    "NOX": "Concentración de óxidos de nitrógeno (ppm) (numérico real)",
    "RM": "Número promedio de habitaciones (numérico real)",
    "AGE": "Proporción de viviendas construidas antes de 1940 (numérico real)",
    "DIS": "Distancia a centros de empleo (numérico real)",
    "RAD": "Índice de accesibilidad a autopistas radiales (categórica ordinal)",
    "TAX": "Tasa de impuestos sobre propiedades (numérico entero)",
    "PTRATIO": "Relación alumno-profesor (numérico real)",
    "B": "Proporción de personas de origen afroamericano (numérico real)",
    "LSTAT": "Porcentaje de población de bajos ingresos (numérico real)"
}

# Categorías disponibles para las variables categóricas
chas_options = [0, 1]  # CHAS solo puede ser 0 o 1
rad_options = list(range(1, 25))  # Suponiendo que RAD es un índice con valores entre 1 y 24

# Crear una función que construya la interfaz y haga la predicción
def predict_price(model):
    # Título de la app
    st.title('Predicción del valor de una vivienda en Boston')

    if st.checkbox("# Ver análisis del modelo"):
        st.write("#### Análisis de hiperparámetros")
        st.markdown("""La búsqueda del mejor modelo regresión para la estimación del valor medio de una vivienda en Boston, empleando la información recopilada en la base de datos boston_housing, se realizó probando los métodos de ElasticNet y diferentes opciones de KernelRidge. Cabe recordar que el primer caso reúne, a su vez, la posibilidad de probar regresiones Lasso y Ridge, es decir se emplean conjuntamente l1 y l2. La búsqueda del mejor modelos contempla las variables en su dimensión original sin escalar, así como también aplicando técnicas de escalado como normalización (StandarScaler) o redimensionando en función de mínimos y máximos (MinMaxScaler)

Para ElasticNet se probaron valores de l1 desde 0,1 a 1,0 (0.1,0.2,0.5,1.0) para considerar la posibilidad de “apagado” de determinada regresión (Lasso o Ridge), además de diferentes combinaciones para la selección de variables. Igualmente, se probaron valores de alpha que condicionan la dimensión de los coeficientes del modelo para evitar su sobreajuste (0.1,0.2,0.5,1.0,10.0,100.0). En cuanto a KernelRidge se probaron opciones de alpha, que en este caso se traduce en penalización para ajustar el modelo, con valores entre 0,1 y 100 (0.1,0.2,0.5,1.0,10.0,100.0). Adicionalmente se probaron los siguientes  kernels que determinan transformación de datos a espacios de mayor dimensionalidad que facilitan la estimación del modelo de regresión buscado: lineal, polinomial, Radial Basis Fuction (rbf) y sigmoide.

Con esta información el mejor modelo obtenido fue evaluado con el menor valor absoluto de error definido como MAE como una medida del error de predicción expresa en las mismas unidades de la variable objetivo. Se obtiene que el mejor modelo tiene un MAE de 2,6156 miles de dólares, según los resultados en el conjunto de prueba, es decir, que los precios medios de vivienda predichos pueden presentar errores de estimación medios de este valor en ambos sentidos, condición que debe tenerse en cuenta al momento de realizar las interpretaciones respectivas. 

Este mejor modelo de regresión encontrado corresponde al método KernelRdige, específicamente empleando un kernel gaussiano RBF (Radial Basis Function) o también denominado exponencial que permite llevar los datos sucesivamente a espacios o dimensiones mayores y de esta forma optimizar la estimación del modelo de regresión, particularmente facilitando la interpretabilidad de las variables. 

La obtención de este modelo parte de variables escaladas como se observa del hiperparámetro correspondiente, StandarScaler, que indica que normalizar las variables, e decir, llevarlas a un rango de 0 a 1, permite, por una parte, controlar la dimensionalidad evitando impactos excesivos de alguna de las variables y, por otro lado, encontrar mejores coeficientes para predicciones más acertadas. 

El alpha obtenido fue de 0,1 el cual es el valor más bajo entre las opciones probadas y significa una penalización muy baja; en otras palabras, puede decirse que el modelo ignora pocos valores en el entrenamiento y por lo tanto puede ajustarse mejor a los dato observados, aunque con riesgo de sobreajuste.

Por último, se advierte que se presentan los hiperparámetros, que no fueron probados, de coeficiente (coef) y grado (degree), pero dado que el mejor modelo obtenido corresponde un kernel rbf, estos conceptos pierden relevancia para la estimación y pueden ignorarse.
""")
    
    # Explicación breve
    st.write("#### Datos de la vivienda")
    st.write("Introduce los datos de la vivienda para estimar su precio promedio.")
    
    # Inicializar las variables en st.session_state si no están presentes
    for col in columns:
        if f"input_{col}" not in st.session_state:
            st.session_state[f"input_{col}"] = 0.0  # Inicializar cada variable individualmente con valor 0.0

    # Organizar la entrada en forma de tabla con 6 columnas
    input_data = {}

    # Número de columnas que queremos
    num_columns = 3

    # Crear número de filas necesario según el número de variables
    for i in range(0, len(columns), num_columns):
        # Crear 3 columnas para cada fila
        cols = st.columns(num_columns)
        
        for j, col in enumerate(columns[i:i+num_columns]):
            # Usamos un selectbox para CHAS y RAD, y text_input para el resto de las variables
            if col == "CHAS":
                input_value = cols[j].selectbox(
                    f"Ingrese el valor para {col} (0 o 1)", 
                    options=chas_options,
                    help=column_types[col]
                )
            elif col == "RAD":
                input_value = cols[j].selectbox(
                    f"Ingrese el valor para {col} (1-24)", 
                    options=rad_options,
                    help=column_types[col]
                )
            else:
                # Para otras variables numéricas, seguimos usando text_input
                input_value = cols[j].text_input(
                    f"Ingrese el valor para {col}",
                    value=str(st.session_state[f'input_{col}']),  # Como texto para evitar botones
                    help=column_types[col]  # Mostrar el tipo de la variable
                )

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
        st.write(f"### El valor estimado de la vivienda es: ${prediction[0]:,.4f}")

    # Mostrar los hiperparámetros si el checkbox está marcado
    if st.checkbox("Mostrar hiperparámetros del modelo"):
        st.write("#### Hiperparámetros del modelo")
        # Obtener los hiperparámetros del modelo
        model_params = model.get_params()  # Para un modelo de sklearn, como LinearRegression
        
        # Convertir los hiperparámetros a un DataFrame para mostrarlos de manera ordenada
        df_params = pd.DataFrame(list(model_params.items()), columns=['Hiperparámetro', 'Valor'])

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
        
        # Mostrar la tabla con los hiperparámetros
        st.write(df_params.to_html(index=False, escape=False), unsafe_allow_html=True)


def main():
    # Cargar el modelo
    model = load_model()

    # Ejecutar la predicción
    predict_price(model)


# Si el script es ejecutado directamente, se llama a main()
if __name__ == "__main__":
    main()
