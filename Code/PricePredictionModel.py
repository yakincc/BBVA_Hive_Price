import pickle
import gzip
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def asignar_encoder_nuevo(df, code_columns, mapeos):
    """
    Asigna un entero a cada valor de cada columna de un dataframe utilizando
    mapeos previamente asignados, o asignando nuevos enteros si el valor es desconocido.
    Los mapeos de valores a enteros se almacenan en un diccionario.
    
    Parameters:
    df (pandas.DataFrame): El dataframe al que se le asignarán los enteros.
    code_columns: Las columnas del dataframe sobre las que se hará la codificación
    mapeos (dict): Un diccionario con los mapeos de valores a enteros para cada columna.
    
    Returns:
    pandas.DataFrame: El dataframe con las columnas convertidas a enteros.
    dict: Un diccionario actualizado con los mapeos de valores a enteros para cada columna.
    """
    for col in code_columns:
        if col in mapeos:
            # Si la columna está en los mapeos, se utiliza el mapeo existente
            categorias = mapeos[col]
        else:
            # Si la columna no está en los mapeos, se asignan enteros nuevos
            categorias = {}
        # Se crea una función lambda para aplicar a cada fila del dataframe
        mapper = lambda x: categorias[x] if x in categorias else categorias.setdefault(x, max(categorias.values(), default=-1) + 1)
        # Se aplica la función a la columna correspondiente del dataframe
        df[col] = df[col].map(mapper)
        # Se actualizan los mapeos para la columna correspondiente
        mapeos[col] = categorias
    return df, mapeos

def preprocess(data, get_encoders = False, encoders_dictionary = {}, previous_cleaning = True):
    """
    Función que preprocesa un conjunto de datos de bienes raíces.
    Args:
    data (DataFrame): Conjunto de datos de bienes raíces a preprocesar.
    Returns:
    dataset (DataFrame): Conjunto de datos preprocesado.
    """
    dataset = data.copy()
    #Eliminación de las características irrelevantes para el análisis
    try:
        dataset = dataset.drop(['Número de frentes','Número de estacionamiento', 'Elevador', 'Piso', 'Depósitos', 'Posición', 'Método Representado', 'Estado de conservación'], axis=1)
    except:
        pass
    
    dataset['Categoría del bien'] = dataset ['Categoría del bien'].fillna('Missing')
    
    #Homogeneización para mantener una única moneda (USD)
    try:
        dataset.loc[dataset['Moneda principal para cálculos']=='PEN','Valor comercial'] *=0.26
        dataset = dataset.drop('Moneda principal para cálculos', axis=1)
    except:
        dataset = dataset.drop('Moneda principal para cálculos', axis=1)

    #Conversión del año de valorización a formato de fecha
    dataset['Año'] = dataset['Fecha entrega del Informe'].dt.strftime('%Y')
    dataset = dataset.drop('Fecha entrega del Informe', axis= 1)
                 
    #Aignación de valores numéricos adecuados
    dataset[['Área Terreno', 'Área Construcción', 'Latitud (Decimal)', 'Longitud (Decimal)']] = dataset[['Área Terreno', 'Área Construcción', 'Latitud (Decimal)', 'Longitud (Decimal)']].astype(float)
      
    #Codificación de las variables categóricas
    columns = ['Categoría del bien', 'Calle', 'Provincia', 'Distrito', 'Departamento']
    dataset, encoders_dictionary = asignar_encoder_nuevo(dataset, columns, encoders_dictionary)
    
    with open("encoders.json", "w") as f:
        json.dump(encoders_dictionary, f)
    
    #Rellenado de valores nulos
    dataset[['Distrito', 'Provincia', 'Departamento']] = dataset[['Distrito', 'Provincia', 'Departamento']].fillna(dataset[['Distrito', 'Provincia', 'Departamento']].mode().iloc[0])
    dataset[['Edad', 'Área Construcción']] = dataset[['Edad', 'Área Construcción']].fillna(0)
    dataset['Área Terreno'] = dataset ['Área Terreno'].fillna(dataset['Área Terreno'].mean())
    
    #Limpieza de datos atípicos ilógicos
    if previous_cleaning:
        dataset = dataset.query(' `Latitud (Decimal)` < -0.03 & `Latitud (Decimal)` > -18.3522222 & `Longitud (Decimal)` > -81.32638888888889 & `Longitud (Decimal)` < -68.6575')
        dataset = dataset[dataset['Edad'] <= 300]

    #Método de imputación a valores vacíos de Latitud y Longitud
    Distritos2 = dataset['Distrito'].unique()
    Distritos2.shape
    DicLon = {}
    DicLat = {}
    for i in range(len(Distritos2)):
        Distritoi = dataset.loc[dataset['Distrito'] == Distritos2[i]]
        a = Distritoi['Latitud (Decimal)'].mean()
        b = Distritoi['Longitud (Decimal)'].mean()
        DicLat[Distritos2[i]]=round(a, 6)
        DicLon[Distritos2[i]]=round(b, 6)
        
    dataset.loc[dataset['Longitud (Decimal)'].isnull(), 'Longitud (Decimal)'] = dataset['Distrito'].map(DicLon)
    dataset.loc[dataset['Latitud (Decimal)'].isnull(), 'Latitud (Decimal)'] = dataset['Distrito'].map(DicLat)
    dataset = dataset.reset_index(drop = True)
    
    if get_encoders:
        return dataset, encoders_dictionary
    else:
        return dataset

def deteccion_outliers(data, columnas, upq = 0.98, lwq = 0.02):
    '''
    Elimina los valores atípicos por columna, valores de la columna demasiado alejados del centro, que podrían deberse a errores a la hora de introducirlos a la base de datos.
    
    Args:
    data (pandas.DataFrame): Dataframe con los datos que se desea analizar.
    columnas (list): Lista de strings con los nombres de las columnas en las que se desea eliminar los outliers.
    upq (float): Cuantil superior a partir del cual se considera que un valor es un outlier. Por defecto es 0.98.
    lwq (float): Cuantil inferior a partir del cual se considera que un valor es un outlier. Por defecto es 0.02.
    
    Returns:
    pandas.DataFrame: Dataframe con los outliers eliminados en las columnas especificadas.
    '''
    
    # Se realiza una copia del dataframe original para no modificarlo
    dataset = data.copy()
    
    # Se itera sobre las columnas especificadas para detectar y eliminar outliers
    for columna in columnas:
        # Se calculan los cuantiles superior e inferior para la columna actual
        upper = dataset[columna].quantile(upq)
        lower = dataset[columna].quantile(lwq)
        
        # Se utiliza una función lambda para aplicar la condición de que los valores deben estar dentro del rango de cuantiles
        # Se crea un dataframe temporal con los valores que cumplen esa condición
        df = dataset.apply(lambda row: row[(dataset[columna] <= upper) & (dataset[columna] >= lower)])
    
    # Se devuelve el dataframe con los outliers eliminados
    return df

class PricePredictionModel():
    def __init__(self, n_estimators=100, min_samples_split=2, min_samples_leaf = 2, random_state = 2, n_jobs = -1):
        '''
        Inicialización de un modelo de Árboles de Decisión. Por defecto:
        :param n_estimators: (int) Número de árboles de decisión que se construyen en el bosque aleatorio
        :param min_samples_split: (int) Cantidad mínima de muestras que se requieren para dividir un nodo interno en un árbol de decisión.
        '''
        self.model = RandomForestRegressor(min_samples_split = min_samples_split, n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, random_state = random_state, n_jobs = n_jobs)
        

    def save_model(self, file_path):
        """
        Guarda el modelo entrenado utilizando la librería pickle.

        Parámetros:
        -----------
        file_path: str
            Ruta donde se guardará el modelo entrenado.
        """
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_model(self, file_path):
        """
        Abre el modelo ya entrenado desde el archivo file_path.

        Parámetros:
        -----------
        file_path: str
            Ruta donde se encuentra el modelo entrenado.
        """
        with gzip.open(file_path, 'rb') as f:
            self.model = pickle.load(f)
            
    def pipeline(self, X, model_path, encoder_path, train_data_path = None, data_preprocess=True, return_y = True, retrain=False, xlsx_output=False, output_path='TEST_PREDICTIONS.xlsx', save_model_path = 'Hive_price.pkl.gz'):
        """
        Realiza el pipeline completo de entrenamiento, predicción y guardado del modelo.

        Parámetros:
        -----------
        X: pd.DataFrame
            Datos de entrada para entrenar o predecir.
        model_path: str
            Ruta donde se guardará el modelo entrenado o donde se encuentra el modelo entrenado previamente.
        encoder_path: str
            Ruta donde se encuentra el archivo con los encoders utilizados para la preprocesamiento de los datos.
        train_data_path: str
            Ruta donde se encuentran los datos con que se entrenó el modelo. Sólo si se necesita hacer un reentrenamiento.
        data_preprocess: bool, optional (default=True)
            Indica si se debe realizar el preprocesamiento de los datos antes de hacer la predicción.
        return_y: bool, optional (default=True)
            Indica si se desea obtener el resultado de la predicción como return del método pipeline.
        retrain: bool, optional (default=False)
            Indica si se debe reentrenar el modelo con los datos de entrada.
        xlsx_output: bool, optional (default=False)
            Indica si se deben guardar las predicciones en un archivo xlsx.
        output_path: str, optional (default='TEST_PREDICTIONS.xlsx')
            Ruta donde se guardará el archivo xlsx con las predicciones.

        Retorna:
        -----------
        y_pred: np.array
            Array con las predicciones del modelo.
        """
        X_clean = X
        if data_preprocess:
            # Realiza el preprocesamiento de los datos utilizando los encoders guardados en encoder_path
            with open(encoder_path, "r") as f:
                encoders_dictionary = json.load(f)
            X_clean = preprocess(X, previous_cleaning=False, encoders_dictionary=encoders_dictionary)
            
        # Carga el modelo desde el archivo model_path
        self.load_model(model_path)
        
        # Realiza las predicciones utilizando el modelo cargado
        y_pred = self.predict(X_clean, xlsx_output=xlsx_output, file_path=output_path)
        y_pred = pd.DataFrame(y_pred, columns = ['Valor comercial (Modelo)'])
        y_pred = y_pred.reset_index(drop = True)
        
        if retrain:
            # Se cargan los datos originales desde el archivo xlsx en train_data_path
            X_train_orig = pd.read_excel(train_data_path,  header=0, thousands=",")
            y_train_orig = pd.DataFrame(X_train_orig['Valor comercial'].copy(), columns = ['Valor comercial'])
            
            X_train_new = pd.concat([X_train_orig, X], axis=0).reset_index(drop = True)
            X_train_new = X_train_new.drop('Valor comercial', axis = 1)
            
            y_pred = y_pred.rename(columns = {'Valor comercial (Modelo)' : 'Valor comercial'})
            y_train_new = pd.concat([y_train_orig, y_pred], axis=0).reset_index(drop = True)
            
            #Se guarda el nuevo dataset de entrenamiento en la ruta: 'DATA_RETRAIN.xlsx' para su uso posterior.
            new_train_data = pd.concat([X_train_new, y_train_new], axis = 1) 
            new_train_data.to_excel('DATA_RETRAIN.xlsx')
            
            #Se preprocesan los datos para poderse utilizar en el modelo
            X_clean_new = preprocess(X_train_new, previous_cleaning=False, encoders_dictionary=encoders_dictionary)
            
            # Se reentrena el modelo con los datos originales y los nuevos datos concatenados
            self.fit(X_clean_new, y_train_new)

            # Guarda el modelo reentrenado en model_path
            self.save_model(save_model_path)
        
        if return_y:
            return y_pred

    def fit(self, X_train, y_train):
        '''
        Entrena el modelo de predicción de precios.
        :param X_train: (pd.DataFrame) Conjunto de datos para el entrenamiento.
        :param y_train: (pd.DataFrame) Variable objetivo para la regresión
        '''
        self.model.fit(X_train, y_train)

    def predict(self, X_test, xlsx_output = False, file_path = 'TEST_PREDICTIONS.xlsx'):
        '''
        Predicción del precio para los datos requeridos.
        :param X_test: (pd.DataFrame) Conjunto de datos sobre los que predecir el valor comercial
        :param xlsx_output: (Bool) Si se desea, o no, guardar los resultados en un archivo xlsx
        :return: (list) Lista de valores predichos para cada muestra en X_test
        '''
        y_pred = self.model.predict(X_test)
        y_pred = [int(y) for y in y_pred]

        if xlsx_output:
            df = pd.DataFrame(y_pred, columns=['Valor comercial'])
            df.index = np.arange(1, len(df) + 1)
            df.to_excel(file_path)
            
        return y_pred
