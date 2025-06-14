from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

def train_random_forest(X_train, y_train):
    """
    Crea, entrena y ajusta un Random Forest con GridSearchCV.
    Incluye texto (TF-IDF), categoría (OneHot) y longitud del título (escalado).
    """
    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('texto', TfidfVectorizer(max_features=5000), X_train.columns[0]),
            ('categoria', OneHotEncoder(), ['category'])
        ]
    )

    # Pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Grid de parámetros
    param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__min_samples_split': [2, 5]
    }

    # GridSearchCV
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Mejores parámetros encontrados:", grid.best_params_)

    return grid.best_estimator_


def train_SVM(X_train, y_train):
    """
    Crea y entrena un pipeline de SVM con preprocesamiento
    para texto, categoría y longitud del título.
    """
    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('texto', TfidfVectorizer(max_features=5000), X_train.columns[0]),
            ('categoria', OneHotEncoder(), ['category'])
        ]
    )

    # Pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LinearSVC())
    ])

    param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__max_iter': [3000, 5000, 10000]
    }

    # Entrenar
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Mejores parámetros encontrados:", grid.best_params_)
    print("Mejor accuracy promedio (cv=5):", grid.best_score_)

    return grid.best_estimator_




def train_naive_bayes(X_train, y_train):
    """
    Entrena un pipeline de Naive Bayes con preprocesamiento:
    - Bolsa de palabras con CountVectorizer para el texto
    - OneHotEncoder para categoría
    - Requiere que la longitud del título esté discretizada si se incluye

    Parámetros:
    - X_train: DataFrame con columnas 'sentences', 'category', 'title_length_bin' (opcional)
    - y_train: etiquetas

    Devuelve:
    - pipeline entrenado
    """  

    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('texto', TfidfVectorizer(max_features=5000), X_train.columns[0]),
            ('categoria', OneHotEncoder(), ['category'])
        ]
    )

    # Pipeline final
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', MultinomialNB())
    ])
    
    param_grid = {
        'clf__alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
        'clf__fit_prior': [True, False],
    } 
    # Entrenar
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Mejores parámetros encontrados:", grid.best_params_)
    print("Mejor accuracy promedio (cv=5):", grid.best_score_)

    return grid.best_estimator_

