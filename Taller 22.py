import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import fetch_california_housing
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
from sklearn.impute import KNNImputer
from ipywidgets import interact, Dropdown
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Título de la app
st.title("Taller 1 Minería de datos")
st.write("Oscar Leonardo Herrera Correa")


# Texto introductorio
st.write("### Análisis de base de datos")

bases= ["",'Titanic', 'California']

st.sidebar.header('Opciones')
base_seleccionada = st.sidebar.selectbox('# Selecciona una base de datos:', bases)

base_trabajo=()
if base_seleccionada == 'Titanic':
  base_trabajo=sns.load_dataset('titanic')
elif base_seleccionada == 'California':
  base_trabajo=fetch_california_housing(as_frame=True).frame
else:
  st.write("No ha seleccionado una base de datos")

if st.sidebar.checkbox("## Ver análisis de resultados"):
  if base_seleccionada == 'Titanic':
    st.write("### Análisis de resultados base de datos Titanic")
    st.markdown(""" La información de la base de datos Titanic indica que ésta tiene un total de 15 variables que se distribuyen por tipos de la siguiente manera: bool (2), category (2), float64 (2), int64(4), object (5). Estas características indican una alta diversidad de la información disponible. En este punto, debe advertirse que variables como age, embarked, deck y embar_town presentan datos faltantes o nulos frente a los 891 registros totales de la base de datos, por lo cual es necesario dar un tratamiento a los mismos. La información se presenta para las primeras filas del dataset, dada la extensión del mismo.

Las estadísticas descriptivas, calculadas para las variables de tipo numérico, permite aproximarse al entendimiento de la dimensionalidad de las variables, así como confirmar la naturaleza de las mismas. De acuerdo con la información obtenida, es posible señalar que las variables survived y pclas, son, en realidad, categóricas y no numéricas, según su rango, así como las frecuencias por variables analizadas posteriormente, por lo cual se brinda la opción de reclasificarlas para los procesos siguientes. Las restantes clasificadas como int o float sí tienen naturaleza numérica, con rangos entre 0,42 y 80 años para la primera y entre 0 y 512.3292 (moneda por determinar) para la segunda, con medias de 29,6991 años para los pasajeros que abordaron el barco que a su vez pagaron un promedio de 32,2042 (moneda por determinar). Los rangos de sibsp y parch son de mínimo 0 y máximo de 8 y 6 respectivamente.

Dada la tipología de variables, se generan gráficos interactivos para explorar diferentes relaciones. Los gráficos de dispersión e histograma, según lo anotado anteriormente, es aplicables solo para las variables numéricas, Los resultados de este gráfico muestran 2 puntos aparentemente atípicos en edades entre 30 y 40 años que pagaron tarifas superiores a 500 (moneda por determinar). En relación con el resto de puntos, no se percibe un comportamiento lineal, aunque sí una mayor concentración de puntos en tarifas bajas para las diferentes edades, condición esta última, que se percibe de una manera más clara en el histograma cuya distribución se asimila a una chi-cuadrada o F en lugar de una normal. El histograma de age, por el contrario, sugiere tener una distribución normal. Los gráficos de las variables sibsp y parch indican concentración en valores bajos de estas variables en cualquier comparación y dado que son enteros la forma señala linealidad marcada por la dimensión de la variables contra la cual se contrasta.

La relación de familiares evaluada, sibsp y parch, evaluada con edad no permite visualizar tendencias, lo que indica comportamientos aleatorios, similar a la relación con fare, puesto que aunque se observa que el nivel inferior del rango crece con mayor número de familiares, los resultados no terminan siendo homogéneos, por la diversidad de rangos evidenciados.

El gráfico box-plot, permite visualizar comportamientos de variables numéricas y categóricas. En este caso se presentan algunos combinaciones que explican las características de los pasajeros. El análisis de las variables sex y age, indican que los hombres son ligeramente mayores que las mujeres y presentan más datos atípicos por encima del límite superior. Entre sex y fare, se percibe una mayor variación para las mujeres, mientras que en el caso de los hombres existen más datos atípicos en relación a la tarifa. En cuanto a la pclass y class por age, los resultados muestran que a menor edad, la clase ocupada es igualmente menor (3).

Las variables embarked y embark town con fare muestran que en Cherbourg hubo mayor variedad de tarifas pagadas mientras que en Southamptoun y Queenstown fueron más bajas y con menor variabilidad. Por su parte, la relación de deck con age, muestra que personas más jóvenes prefirieron las cubiertas F y G es decir las inferiores, cuyas tarifas, según la variable fare, son más baja.

Se analiza también alive y age encontrándose muy leves diferencias, particularmente, en los límites superiores del rango y de la mayor concentración (percentil 75) límite superior, que es menor para quienes si sobrevivieron. Respecto a fare, los que no sobrevivieron tuvieron en general, tarifas más concentradas en rangos inferiores en relación con los que sí sobrevivieron cuya variabilidad fue mayor.

Analizando la matriz de correlación, se observa que es baja entre age y fare, pues solo alcanza 0,1, es decir muy próxima a ser inexistente, lo que es consistente con el análisis del gráfico de dispersión. Por su parte, la relación más alta se presenta entre fare y pclass, lo cual denota que las diferencias entre clases condicionan los cobros, en este caso de forma inversa, dado que la mejor clase se codifica en el número menor, es decir, 1.

Para las variables con datos faltantes o nulos, se brinda la opción de imputación, para lo cual se recomiendan los siguientes métodos considerando la naturaleza de las mismas. Se propone aplicar media para la variable age puesto que existen cantidad de datos que, como indican un comportamiento normal. Para embarked y embar_town se sugiere eliminar filas puesto que sólo faltan 2 registros y su eliminación no afectaría el análisis, mientras que para deck, la opción sería eliminar la columna dado que sólo se dispone de 203 datos lo cual es un valor muy bajo (inferior al 25% del total de la muestra) por lo cual no aporta mucha información.

Dadas las variables categóricas, es pertinentes codificarlas, para lo cual se recomiendan las siguientes estrategias: Ordinal Encoder para las variables pclass, class, embarked, who, adult_male, embark_town y alone, mientras que OneHot Encoder para sex, alive. En el primer caso, se facilita así el análisis en razón a que se dispone de varias categorías, en algunos casos ordenables jerárquicamente (pclas y class), evitando también extender la base de datos, buscando parsimonia. El motivo de codificar con OneHot Encoder las variables sex y alive responde a la posibilidad de profundizar análisis que determinen relaciones o probabilidades según sus valores, por ejemplo, en posteriores regresiones logísticas.

Las variables numéricas son susceptibles de escalar, para lo cual se recomienda para esta base de datos, la estrategia Standar Scaler, dado que los rangos y dimensiones difieren y con ello se logra reducir el impacto de las variables sobre análisis globales y a su vez de los datos atípicos.
""")
  elif base_seleccionada == 'California':
      st.write("### Análisis de resultados base de datos California")
      st.markdown(""" La información de la base de datos California contiene un total de 9 todas ellas numéricas. La base reúne un total de 20640 registros sin presentar datos faltantes o nulos por variable. La información se presenta para las primeras filas del dataset, dada la extensión del mismo.

Las estadísticas descriptivas permiten aproximarse al entendimiento de la dimensionalidad de las variables, así como confirmar la naturaleza de las mismas. Por ejemplo, mientras el rango de la variable MediaHouseVal está entre 0,15 y 5, la variable Population reporta valores entre 3 y 35.682, lo que señala diferencias importantes en escalas de variables. Como datos relevantes, se reseña la media de la variable HouseAge que es de 28,6395 años de antigüedad de las viviendas. También se referencia la media de MedInc que es de 3,87707 lo que da una idea d los ingresos promedio.

Se generan gráficos interactivos para explorar el comportamiento de las variables. En premire lugar se genera histogramas con los cuales se analiza posibles distribuciones de las variables. Según estos gráficos, las variables MedInc, HouseAge, y MedHouseVal evidencian formas similares a distribuciones normales que orientan futuros análisis y revisiones. Por su parte, las variables, AveRooms, AveBedrms, Population, y AveOccup concentran sus valores en niveles inferiores lo que sugiere comportamientos compatibles con distribuciones chi cuadradas o F. No se grafican Latitude y Longitude dado que, pese a ser numéricas, representan localización geográfica.

Los gráficos de dispersión, por su parte, permiten evidenciar comportamientos o relaciones aleatorias entre algunas de la as variables consideradas, puesto que no se insinúa linealidad entre ellas o tendencia que refleje determinada relación. Es el caso, por ejemplo, de las variables HouseAge con MedInc y MediaHouseVal. Adicionalmente, otras revisiones permiten ver que la variable AveRooms tiende a concentrar sus valores en niveles bajos, comportamiento que condiciona la relación con variables como MedInc, HouseAge,
y MedHouseVal que reflejan también esta concentración casi de forma lineal paralela al eje, lo cual, resulta útil para visualizar 2 datos atípicos de la variable AveRooms. Esta misma variable evidencia un relación directa con AveBedrms derivada posiblemente de la condición estructural de las viviendas. En contraste la AveOccup parece ser constante en función del AveRooms, pues se mantiene en valores cercanos a 0 independiente del número de habitaciones. Adicionalmente, se analiza la variable Population, la cual muestra concentración de valores en niveles bajos de población y en un banda paralela al eje, lo que no permite visualizar relaciones con variables como MedInc, HouseAge y MediaHouseVal, mientras que con AveRooms y AveBedrms se observa una curva parabólica que señala que a mayor población menor número de cuartos promedio, lo que puede estar determinado por la escala y la fórmula de cálculo del promedio de la variable, antes que por un desempeño particular.
En cuanto a los gráficos de box-plot, estos no resultan relevantes ante la ausencia de variables categóricas, por lo cual se omiten en el presente análisis.


Analizando la matriz de correlación, se observa que la relación más alta es entre AveRooms y AveBedrms con 0,85 confirmando el análisis de la sección anterior, y lo que indica que podría omitirse una de ellas porque la explicación que aporta es igual en cualquiera de ellas. A esta correlación, le sigue en magnitud la correlación entre MedInc y MedHouseVal con 0,69, explicable por la posibilidad de acceso a viviendas más costosas por parte de personas con mayores ingresos. En contraste, las correlaciones más bajas, cercanas a cero (0), corresponde a la variable AveOccup con las restantes, lo que indica que no se ve afectada por el comportamiento de las demás, o en otras palabras es independiente.

Dado que no hay datos faltantes o nulos no se hace necesario aplicar estrategias de imputación. Asimismo, la ausencia de variables categóricas hace prescindir de la posibilidad de codificarlas, por lo cual se omiten el empleo de estas opciones del menú.

La condición de que todas las variables son numéricas y a su vez presentan diferencias de escala significativas como se anotó en sección anterior, hace que sea conveniente escalar los valores para lo cual se se recomienda, para esta base de datos, la estrategia Standar Scaler, buscando reducir el impacto de las variables sobre análisis de globales e igualmente de los datos atípicos.
""")

base_trabajo=pd.DataFrame(base_trabajo)

st.sidebar.header("Exploración de datos")

if st.sidebar.checkbox("Mostrar primeras filas"):
    n_rows = st.sidebar.slider("Número de filas a mostrar:", 1, 50, 5)
    st.write(f"### Primeras {n_rows} filas del dataset")
    st.write(base_trabajo.head(n_rows))

if st.sidebar.checkbox("Mostrar información del dataset"):
    st.write("### Información del dataset")

    buffer = io.StringIO()
    base_trabajo.info(buf=buffer)

    info_text = buffer.getvalue()
    st.text(info_text)

if st.sidebar.checkbox("Mostrar estadísticas descriptivas"):
    st.write("### Estadísticas descriptivas")
    st.write(base_trabajo.describe())


if st.sidebar.checkbox("Mostrar frecuencias por variables"):
    variable1 = st.sidebar.selectbox('Selecciones una variable:', base_trabajo.columns)
    if variable1:
        st.write("### Frecuencias por variable")
        st.write(base_trabajo[variable1].value_counts())

if st.sidebar.checkbox("Mostrar información por registro"):
    num_fila = st.sidebar.number_input("Ingresa el número de registro", value=0)
    if num_fila >= 0 and num_fila < len(base_trabajo):
      st.write("### Información por registro")
      st.write(base_trabajo.iloc[num_fila][base_trabajo.columns])
    else:
      st.write("El número de registro ingresado no es válido")

st.sidebar.header("Reclasificación de variables")
if st.sidebar.checkbox("Reclasificar variables numéricas"):
  st.sidebar.write("Variables numéricas")
  for col1 in base_trabajo.columns:
     if base_trabajo[col1].dtype in ['int64', 'float64']:
       if st.sidebar.checkbox(f"Reclasificar {col1}"):
          base_trabajo[col1] = base_trabajo[col1].astype('category')

if st.sidebar.button("Mostrar información del dataset reclasificado"):
   st.write("## Información del dataset reclasificado")
   buffer2 = io.StringIO()
   base_trabajo.info(buf=buffer2)
   info_text2 = buffer2.getvalue()
   st.text(info_text2)

# Sección para gráficos dinámicos

st.sidebar.header("Gráficos dinámicos")
if st.sidebar.checkbox("Generar gráficos"):

# Selección de variables para el gráfico
  x_var = st.sidebar.selectbox("Selecciona la variable X:", base_trabajo.columns)
  y_var = st.sidebar.selectbox("Selecciona la variable Y:", base_trabajo.columns)

  # Tipo de gráfico
  chart_type = st.sidebar.radio(
      "Selecciona el tipo de gráfico:",
      ("Dispersión", "Histograma", "Boxplot")
  )

# Mostrar el gráfico

  st.write("### Gráficos")
  if chart_type == "Dispersión":
      st.write(f"#### Gráfico de dispersión: {x_var} vs {y_var}")
      fig, ax = plt.subplots()
      sns.scatterplot(data=base_trabajo, x=x_var, y=y_var, ax=ax)
      st.pyplot(fig)
  elif chart_type == "Histograma":
      st.write(f"#### Histograma de {x_var}")
      fig, ax = plt.subplots()
      sns.histplot(base_trabajo[x_var], bins=30, kde=True, ax=ax)
      st.pyplot(fig)
  elif chart_type == "Boxplot":
      st.write(f"#### Boxplot de {y_var} por {x_var}")
      fig, ax = plt.subplots()
      sns.boxplot(data=base_trabajo, x=x_var, y=y_var, ax=ax)
      st.pyplot(fig)

st.sidebar.header("Matriz de correlación")

if st.sidebar.checkbox("Matriz de correlación"):
    base_trabajo_c = base_trabajo.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = base_trabajo_c.corr()
    st.write("## Matriz de correlación:")
    st.write(correlation_matrix)
    st.write("Gráfico de Correlación:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(plt)



st.sidebar.header("Ajuste de datos")

# Copiar el DataFrame para evitar modificar el original
base_trabajo_copy = base_trabajo.copy()

# Estrategias disponibles
estrategias = ['Media', 'Mediana', 'Moda', 'Eliminar filas', 'Eliminar columna', 'KNN Imputación - numerica']
estrategias1 = ['Moda', 'Eliminar filas', 'Eliminar columna', 'KNN Imputación - no numérica']

# Crear widgets para seleccionar estrategias

knn_imputer = KNNImputer(n_neighbors=5)
estrategia={}
base_trabajo_copy2= base_trabajo_copy.copy()

if st.sidebar.checkbox("Ajustar datos"):
  for col in base_trabajo_copy.columns:
    estrategia[col] = ""
    if base_trabajo_copy[col].isnull().any():
        if base_trabajo_copy[col].dtype in ['int64', 'float64']:
          estrategia[col] = st.sidebar.selectbox(f'Seleccione una estrategia para la variable {col}', estrategias)
        else:
          estrategia[col] = st.sidebar.selectbox(f'Seleccione una estrategia para la variable {col}', estrategias1)

        if estrategia[col]== 'Media':
          base_trabajo_copy2[col].fillna(base_trabajo_copy[col].mean(), inplace=True)
        elif estrategia[col] == 'Mediana':
          base_trabajo_copy2[col].fillna(base_trabajo_copy[col].median(), inplace=True)
        elif estrategia[col] == 'Moda':
          base_trabajo_copy2[col].fillna(base_trabajo_copy[col].mode()[0], inplace=True)
        elif estrategia[col] == 'Eliminar filas':
          base_trabajo_copy2.dropna(subset=[col], inplace=True)
        elif estrategia[col] == 'Eliminar columna':
          base_trabajo_copy2.drop(columns=[col], inplace=True)
        elif estrategia[col] == 'KNN Imputación - numerica':
          base_trabajo_copy2[[col]] = knn_imputer.fit_transform(base_trabajo_copy[[col]])
        elif estrategia[col] == 'KNN Imputación - no numerica':
          base_trabajo_copy2[col] = base_trabajo_copy2[col].fillna('U')

  if st.sidebar.button("Selección finalizada"):
      st.write("## Base de datos resultante")
      buffer2 = io.StringIO()
      base_trabajo_copy2.info(buf=buffer2)
      info_text = buffer2.getvalue()
      st.text(info_text)
      st.write(base_trabajo_copy2.head())




class CategoricalEncodingVisualizer:
    def __init__(self, data):
        self.original_data = data  # Guardar el DataFrame original
        self.categorical_cols = data.select_dtypes(include=['object']).columns  # Columnas categóricas
        self.last_encoded_data = None  # Para almacenar los datos codificados

    def apply_encoding(self, strategy, columns, data_copy):
        """Aplica la estrategia de codificación seleccionada y devuelve una copia codificada."""
        if not len(columns):
            st.write("No hay columnas categóricas seleccionadas.")
            return data_copy

        if strategy == 'Ordinal Encoder':
            encoder = OrdinalEncoder()
            data_copy[columns] = encoder.fit_transform(data_copy[columns])
        elif strategy == 'OneHot Encoder':
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = pd.DataFrame(encoder.fit_transform(data_copy[columns]),
                                        columns=encoder.get_feature_names_out(columns),
                                        index=data_copy.index)
            data_copy = data_copy.drop(columns, axis=1)
            data_copy = pd.concat([data_copy, encoded_data], axis=1)
        else:
            st.write(f"Estrategia desconocida: {strategy}")

        # Guardar la última transformación
        self.last_encoded_data = data_copy
        return data_copy

    def get_last_encoded_data(self):
        """Devuelve los datos codificados de la última transformación."""
        if self.last_encoded_data is not None:
            return self.last_encoded_data
        else:
            st.write("No se ha aplicado ninguna codificación todavía.")
            return None

    def show_encoded_data(self, strategy, columns, data_copy):
        """Muestra una vista previa de los datos codificados usando la estrategia seleccionada."""
        try:
            st.write(f"\nVista previa de los datos codificados usando '{strategy}':")
            encoded_data = self.apply_encoding(strategy, columns, data_copy)
            st.write("Información de los datos codificados:")
            st.dataframe(encoded_data.head())  # Muestra los primeros 5 registros
            st.write(encoded_data.info())  # Muestra la información del DataFrame
        except Exception as e:
            st.write(f"Error al aplicar la estrategia '{strategy}': {e}")


# Código para interactuar con Streamlit y aplicar la clase a los datos
def app():
    # Verificar si 'base_trabajo_copy2' existe y es un DataFrame
    if 'base_trabajo_copy2' in globals() and isinstance(base_trabajo_copy2, pd.DataFrame):
        original_data = base_trabajo_copy2
    else:
        # Si 'base_trabajo_copy2' no existe o no es un DataFrame, usa 'base_trabajo'
        original_data = base_trabajo

    # Instanciamos el visualizador
    visualizer = CategoricalEncodingVisualizer(original_data)

    # Sidebar para seleccionar qué variables codificar
    st.sidebar.header("Codificación de variables categóricas")
    variables_categoricas = []

    # Condición para agregar las variables categóricas a la lista
    if st.sidebar.checkbox("Codificar variables"):
       st.sidebar.write("Variables a codificar")
       for var in original_data.columns:
          if original_data[var].dtype not in ['int64', 'float64']:  # Aseguramos que sean categóricas
             variables_categoricas.append(var)

      # Muestra las variables seleccionadas

    # Diccionario para almacenar las estrategias seleccionadas
    strategy = {}

      # Iteramos sobre las variables categóricas
    for var_cat in variables_categoricas:
          if st.sidebar.checkbox(var_cat):  # Si el checkbox está seleccionado
              # Se despliega el selectbox solo para las variables seleccionadas
              strategy[var_cat] = st.sidebar.selectbox(
                  f'Selecciona una estrategia de codificación para {var_cat}:',
                  ['Ordinal Encoder', 'OneHot Encoder'],
                  index=0  # Puedes cambiar el índice por el valor que prefieras como opción predeterminada
              )

      # Aplicar todas las estrategias seleccionadas y mostrar el DataFrame final
    final_encoded_data = original_data.copy()  # Copia inicial del DataFrame para las transformaciones
    for var_cat, selected_strategy in strategy.items():
          final_encoded_data = visualizer.apply_encoding(selected_strategy, [var_cat], final_encoded_data)  # Acumulando las transformaciones


      # Mostrar el DataFrame codificado final
    if st.sidebar.button("Mostrar base de datos codificada"):
        st.write("# DataFrame con todas las estrategias aplicadas")
        st.session_state.final_encoded_data = final_encoded_data
        st.dataframe(final_encoded_data.head())  # Muestra el DataFrame final con las codificaciones aplicadas



# Ejecutar la app
if __name__ == "__main__":
    app()






class DataScalingVisualizer2:
    def __init__(self, data2):
        self.original_data2 = data2  # Guardar el DataFrame original
        self.numeric_cols2 = base_trabajo.select_dtypes(include=['float64', 'int64']).columns  # Columnas numéricas
        self.last_scaled_data = None  # Para almacenar los datos escalados

    def apply_scaling(self, strategy):
        """Aplica la estrategia de escalado seleccionada y devuelve una copia escalada."""
        data_copy2 = self.original_data2.copy()

        if not len(self.numeric_cols2):
            st.write("No hay columnas numéricas en los datos.")
            return data_copy2

        if strategy == 'Standard Scaler':
            scaler = StandardScaler()
            data_copy2[self.numeric_cols2] = scaler.fit_transform(data_copy2[self.numeric_cols2])
        elif strategy == 'MinMax Scaler':
            scaler = MinMaxScaler()
            data_copy2[self.numeric_cols2] = scaler.fit_transform(data_copy2[self.numeric_cols2])
        elif strategy == 'Robust Scaler':
            scaler = RobustScaler()
            data_copy2[self.numeric_cols2] = scaler.fit_transform(data_copy2[self.numeric_cols2])
        else:
            st.write(f"Estrategia desconocida: {strategy}")

        # Guardar la última transformación
        self.last_scaled_data = data_copy2
        return data_copy2

    def get_last_scaled_data(self):
        """Devuelve los datos escalados de la última transformación."""
        if self.last_scaled_data is not None:
            return self.last_scaled_data
        else:
            st.write("No se ha aplicado ningún escalado todavía.")
            return None

    def show_scaled_data(self, strategy):
        """Muestra una vista previa de los datos escalados usando la estrategia seleccionada."""
        try:
            st.write(f"\nVista previa de los datos escalados usando '{strategy}':")
            scaled_data = self.apply_scaling(strategy)
            st.write(scaled_data.head())  # Mostrar los primeros 5 registros
        except Exception as e:
            st.write(f"Error al aplicar la estrategia '{strategy}': {e}")


# Función principal en Streamlit para cargar y procesar el DataFrame
def app2():
    # Verificar si 'base_trabajo_copy2' existe y es un DataFrame
    if "final_encoded_data" in st.session_state:
        final_encoded_data = st.session_state.final_encoded_data
        original_data2 = final_encoded_data
    elif 'base_trabajo_copy2' in globals() and isinstance(base_trabajo_copy2, pd.DataFrame):
        original_data2 = base_trabajo_copy2
    else:
        # Si 'base_trabajo_copy2' no existe o no es un DataFrame, usa 'base_trabajo'
        original_data2 = base_trabajo

    # Instanciamos el visualizador
    scaler_visualizer = DataScalingVisualizer2(original_data2)

    # Sidebar para seleccionar la estrategia de escalado
    st.sidebar.header("Escalado de variables numéricas")

    # Lista de estrategias
    scaling_strategies = ['Standard Scaler', 'MinMax Scaler', 'Robust Scaler']

    # Selección de la estrategia en el sidebar
    strategy = st.sidebar.selectbox("Selecciona una estrategia de escalado", scaling_strategies, index=0)

    # Mostrar y aplicar la estrategia seleccionada
    if st.sidebar.button('Mostrar base de datos escalada'):
      st.write(f"## DataFrame escalado usando la estrategia: {strategy}")
      scaler_visualizer.show_scaled_data(strategy)

# Ejecutar la app
if __name__ == "__main__":
    app2()




# Mensaje final
st.write("Más opciones en la barra lateral")
