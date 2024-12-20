# Clasificación de Estudiantes
 El conjunto de datos incluye información conocida al momento de la inscripción del estudiante (trayectoria académica, demografía y factores socioeconómicos) y el desempeño académico de los estudiantes. Los datos se utilizan para construir modelos de clasificación para predecir la deserción y el éxito académico de los estudiantes. 
# Proyecto de Clasificación de Estudiantes

## Descripción del Proyecto
Este proyecto tiene como objetivo clasificar a los estudiantes en una de tres categorías:  
- **Reprobado:** Estudiantes que no aprobaron.  
- **Inscrito:** Estudiantes que permanecen inscritos pero no se han graduado.  
- **Graduado:** Estudiantes que completaron exitosamente el programa.  

El análisis se realizó para identificar patrones en los datos y construir modelos de clasificación que permitan predecir la categoría de los estudiantes.


## Objetivo
Desarrollar un modelo de clasificación para predecir el estado final de los estudiantes utilizando información disponible como:
- Estado civil
- Calificación previa
- Género
- Unidades curriculares completadas, entre otros.


## Pasos Realizados

1. **Análisis Exploratorio de Datos (EDA):**
   - Se analizaron las distribuciones de las variables categóricas y numéricas.
   - Se tradujeron todos los nombres de las columnas al español.
     
2. **Preprocesamiento de Datos:**
   - Las variables categóricas (género, estado civil, etc.) fueron codificadas utilizando **LabelEncoder**.
   - Las características numéricas fueron escaladas usando **StandardScaler**.
   - Las clases desbalanceadas fueron tratadas mediante **SMOTE** para generar un dataset balanceado.

3. **Entrenamiento de Modelos:**
   - Se entrenaron múltiples modelos, incluyendo:
     - **Regresión Logística**
     - **Bosques Aleatorios (RandomForest)**
     - **XGBoost**
   - Se realizó una búsqueda de hiperparámetros utilizando **GridSearchCV** para optimizar el modelo XGBoost.
   - Adicionalmente se aplico Validación cruzada estratificada.

4. **Resultados:**
   - Los mejores hiperparámetros encontrados para XGBoost fueron:
     ```
     {
       'colsample_bytree': 0.8,
       'learning_rate': 0.1,
       'max_depth': 3,
       'n_estimators': 200,
       'subsample': 1.0
     }
     ```
   - Precisión final del modelo XGBoost: **86%**.

---

## Requisitos
Las siguientes librerías fueron utilizadas para este proyecto:

```python
import pandas as pd  # Manipulación de datos
import numpy as np  # Operaciones numéricas
import matplotlib.pyplot as plt  # Visualización
import seaborn as sns  # Visualización avanzada
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Preprocesamiento
from sklearn.model_selection import train_test_split, GridSearchCV  # División y búsqueda de hiperparámetros
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Evaluación
from imblearn.over_sampling import SMOTE  # Manejo de desbalance de clases
import xgboost as xgb  # Modelo de clasificación
from sklearn.model_selection import train_test_split, GridSearchCV  # Para dividir datos y búsqueda de hiperparámetros
from sklearn.model_selection import StratifiedKFold # Para el uso de la validación cruzada 



## Uso
1. **Preprocesamiento de Datos:**

```python

# Codificación de variables categóricas
label_encoder = LabelEncoder()
data['Genero_codificado'] = label_encoder.fit_transform(df['Genero'])
data['Estado_civil_codificado'] = label_encoder.fit_transform(df['Estado civil'])

# División del dataset
X = data.drop(['Variable_objetivo'], axis=1)  # Sustituir por el nombre de la variable objetivo
y = data['Variable_objetivo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

2. **Entrenamiento de XGBoost con Hiperparámetros Óptimos:**

```python
xgb_model = xgb.XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=200,
    subsample=1.0,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = xgb_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## Resultados Clave
- El modelo XGBoost mostró un desempeño destacado, alcanzando una precisión del **86%**.
- El uso de **validación cruzada estratificada** mejoró el equilibrio de clases, especialmente para las categorías con menos datos.

  ## AGRADECIMIENTOS
  A los Creadores del datset:
  https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
  
Valentín Realinho Realinho

vrealinho@

Instituto Politécnico de Portalegre

Mónica Vieira Martins Vieira Martins

mvmartins@ipportalegre.pt

Instituto Politécnico de Portalegre

Jorge Machado Machado

jmachado@ipportalegre.pt

Instituto Politécnico de Portalegre

Luis Baptista Bautista

lmtb@ipportalegre.pt

Instituto Politécnico de Portalegre
