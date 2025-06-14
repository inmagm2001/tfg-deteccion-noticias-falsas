# =============================================================================
# DETECTOR DE NOTICIAS Y PUBLICACIONES FALSAS
# =============================================================================
# Aplicación web desarrollada con Shiny para Python que utiliza múltiples
# modelos de inteligencia artificial para detectar contenido falso en 
# noticias y redes sociales. Combina modelos tradicionales de ML, BERT y
# un modelo multimodal para análisis de texto e imagen.
# =============================================================================

# Importaciones principales del framework Shiny y librerías necesarias
from shiny import App, ui, render, reactive
import joblib                    # Para cargar modelos de machine learning tradicionales
from pathlib import Path         # Para manejo de rutas de archivos
import pandas as pd              # Para manipulación de datos
import re                        # Para expresiones regulares en preprocesamiento
import base64                    # Para codificación de imágenes
import traceback                 # Para manejo detallado de errores
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Librerías de procesamiento de lenguaje natural (NLP)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Librerías de deep learning y transformers
import torch
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel
import torch.nn as nn
import numpy as np

# Librerías para procesamiento de imágenes
from PIL import Image
import torchvision.transforms.v2 as v2
from torchvision.models import resnet50, ResNet50_Weights
import io



# =============================================================================
# CONFIGURACIÓN DE PREPROCESAMIENTO DE TEXTO
# =============================================================================
# Configuramos las stopwords en inglés y definimos palabras importantes que 
# NO queremos eliminar (negaciones e intensificadores)

stop_words = set(stopwords.words('english'))
keep_words = {
    'not', 'no', 'never', 'always', 'all', 'every', 
    'very', 'much', 'more', 'most', 'never', 'nothing',
    'everything', 'everyone', 'nobody', 'none'
}
# Removemos las palabras importantes del conjunto de stopwords
stop_words = stop_words - keep_words
lemmatizer = WordNetLemmatizer()

# =============================================================================
# MAPEO DE CATEGORÍAS TEMÁTICAS (K-MEANS CLUSTERING)
# =============================================================================
# Definimos las categorías de noticias identificadas por clustering automático
# Cada cluster representa una temática específica con sus términos característicos




cluster_to_category = {
    0: {
        "name": "gobierno",
        "terms": ["government", "minister", "military", "united", "russia", "iran", "syria", "foreign", "security", "official", "force", "leader"]
    },
    1: {
        "name": "clinton", 
        "terms": ["clinton", "hillary", "email", "campaign", "democratic", "fbi", "election", "candidate", "voter", "investigation", "bernie"]
    },
    2: {
        "name": "corea",
        "terms": ["korea", "north", "missile", "nuclear", "china", "south", "pyongyang", "test", "sanction", "weapon", "seoul"]
    },
    3: {
        "name": "sanidad",
        "terms": ["tax", "healthcare", "obamacare", "congress", "bill", "senate", "legislation", "budget", "plan", "vote", "republican"]
    },
    4: {
        "name": "trump",
        "terms": ["trump", "donald", "president", "campaign", "republican", "twitter", "media", "white", "house", "supporter", "news"]
    },
    5: {
        "name": "sociedad",
        "terms": ["people", "police", "video", "city", "community", "black", "woman", "public", "news", "school", "social"]
    }
}


# =============================================================================
# ETIQUETAS DEL MODELO MULTIMODAL PARA REDES SOCIALES
# =============================================================================
# Mapeo de las clases numéricas del modelo multimodal a etiquetas descriptivas

multimodal_labels = {
    0: "Verdadero",            # Contenido verdadero
    1: "Sátira/Parodia",       # Sátira o parodia
    2: "Engañoso",             # Información que puede inducir a error
    3: "Suplantación",         # Fuente o identidad falsa (impostor)
    4: "Conexión falsa",       # Titular no relacionado con el contenido
    5: "Manipulado"            # Contenido visual o textual alterado
}


# Variable global para almacenar errores durante la carga de modelos
model_loading_errors = []

# =============================================================================
# FUNCIÓN PARA CREAR BARRAS DE CONFIANZA VISUALES
# =============================================================================
def create_confidence_bar(confidence_percent):
    """
    Genera una barra de progreso visual usando caracteres Unicode para mostrar
    el nivel de confianza de las predicciones de manera intuitiva.
    
    Args:
        confidence_percent: Valor de confianza (0-1 o 0-100)
    
    Returns:
        String HTML con la barra de progreso formateada
    """
    # Convertir el porcentaje a valor 0-100 si viene en formato decimal
    if isinstance(confidence_percent, str) and confidence_percent.endswith('%'):
        value = float(confidence_percent.replace('%', ''))
    else:
        value = float(confidence_percent) * 100 if confidence_percent <= 1 else float(confidence_percent)
    
    # Normalizar el valor entre 0 y 100
    value = max(0, min(100, value))
    
    # Configurar la longitud de la barra (30 caracteres)
    bar_length = 30
    filled_length = int((value / 100) * bar_length)
    
    # Caracteres Unicode para la barra visual
    filled_char = "█"    # Carácter lleno
    empty_char = "░"     # Carácter vacío
    
    # Construir la barra visual
    bar = filled_char * filled_length + empty_char * (bar_length - filled_length)
    
    # Retornar HTML con formato estilizado
    return f"{value:.1f}%<br/><span style='font-family: monospace; font-size: 1.1em; letter-spacing: 1px;'>[{bar}]</span>"

# =============================================================================
# FUNCIÓN PARA OBTENER INFORMACIÓN DE CLUSTERS
# =============================================================================
def get_cluster_terms(cluster_id):
    """
    Obtiene el nombre y términos representativos de un cluster específico.
    
    Args:
        cluster_id: ID numérico del cluster
        
    Returns:
        tuple: (nombre_categoria, lista_terminos)
    """
    cluster_info = cluster_to_category.get(cluster_id, {
        "name": "desconocida",
        "terms": None
    })
    return cluster_info["name"], cluster_info["terms"]

# =============================================================================
# FUNCIÓN DE PREPROCESAMIENTO DE TEXTO
# =============================================================================
def preprocess_text(text):
    """
    Aplica una serie completa de transformaciones de preprocesamiento al texto
    para prepararlo para los modelos de machine learning.
    
    Pasos del preprocesamiento:
    1. Conversión a minúsculas
    2. Reemplazo de URLs y emails por tokens
    3. Tokenización de puntuación especial (!, ?, etc.)
    4. Limpieza de caracteres especiales
    5. Tokenización de palabras
    6. Filtrado de stopwords (conservando negaciones)
    7. Lematización
    
    Args:
        text: Texto original a procesar
        
    Returns:
        str: Texto procesado y limpio
    """
    if not isinstance(text, str):
        return ""

    # Paso 1: Convertir a minúsculas
    text = text.lower()
    
    # Paso 2: Reemplazar URLs y emails por tokens especiales
    text = re.sub(r"http\S+|www\S+|https\S+", " urltoken ", text)
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)
    
    # Paso 3: Tokenizar puntuación especial (preservar información emocional)
    text = re.sub(r'!{2,}', ' strongexclamationtoken ', text)      # Múltiples !
    text = re.sub(r'\?{2,}', ' strongquestiontoken ', text)       # Múltiples ?
    text = re.sub(r'[!?]{3,}', ' emphasistoken ', text)           # Énfasis mixto
    text = re.sub(r'!', ' exclamationtoken ', text)               # ! simple
    text = re.sub(r'\?', ' questiontoken ', text)                 # ? simple
    
    # Paso 4: Eliminar todos los caracteres no alfabéticos
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Paso 5: Tokenizar en palabras individuales
    tokens = word_tokenize(text)
    
    # Paso 6: Filtrar stopwords pero conservar palabras importantes
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Paso 7: Unir tokens procesados en texto final
    return ' '.join(tokens)

# =============================================================================
# FUNCIÓN PARA MOSTRAR PASOS DETALLADOS DEL PREPROCESAMIENTO
# =============================================================================
def get_preprocessing_steps(text):
    """
    Ejecuta el preprocesamiento paso a paso y genera un reporte detallado
    de cada transformación aplicada al texto. Útil para explicabilidad.
    
    Args:
        text: Texto original a analizar
        
    Returns:
        list: Lista de strings describiendo cada paso del proceso
    """
    steps = []
    
    # Paso 1: Conversión a minúsculas
    text_lower = text.lower()
    steps.append(f"1. Conversión a minúsculas: {text_lower[:200]}{'...' if len(text_lower) > 200 else ''}")
    
    # Paso 2: Detección y reemplazo de URLs y emails
    text_urls = re.sub(r"http\S+|www\S+|https\S+", " urltoken ", text_lower)
    text_urls = re.sub(r'\S+@\S+', ' emailtoken ', text_urls)
    
    if text_urls != text_lower:
        urls_found = len(re.findall(r"http\S+|www\S+|https\S+|\S+@\S+", text_lower))
        steps.append(f"2. URLs/emails detectados y reemplazados ({urls_found} encontrados): {text_urls[:200]}{'...' if len(text_urls) > 200 else ''}")
        text_lower = text_urls
    else:
        steps.append(f"2. URLs/emails: No se encontraron URLs o emails en el texto")
    
    # Paso 3: Procesamiento de puntuación especial
    original_punct = text_lower
    text_punct = re.sub(r'!{2,}', ' strongexclamationtoken ', text_lower)
    text_punct = re.sub(r'\?{2,}', ' strongquestiontoken ', text_punct)
    text_punct = re.sub(r'[!?]{3,}', ' emphasistoken ', text_punct)
    text_punct = re.sub(r'!', ' exclamationtoken ', text_punct)
    text_punct = re.sub(r'\?', ' questiontoken ', text_punct)
    
    if text_punct != original_punct:
        punct_changes = []
        if '!!' in original_punct: punct_changes.append("exclamaciones múltiples")
        if '??' in original_punct: punct_changes.append("preguntas múltiples")
        if '!' in original_punct: punct_changes.append("exclamaciones simples")
        if '?' in original_punct: punct_changes.append("preguntas simples")
        
        steps.append(f"3. Puntuación especial tokenizada ({', '.join(punct_changes)}): {text_punct[:200]}{'...' if len(text_punct) > 200 else ''}")
        text_lower = text_punct
    else:
        steps.append(f"3. Puntuación especial: No se encontró puntuación especial significativa")
    
    # Paso 4: Limpieza de caracteres especiales
    text_clean = re.sub(r'[^a-z\s]', '', text_lower)
    removed_chars = set(text_lower) - set(text_clean) - {' '}
    
    if removed_chars:
        steps.append(f"4. Caracteres especiales eliminados ({len(removed_chars)} tipos: {', '.join(sorted(removed_chars)[:10])}): {text_clean[:200]}{'...' if len(text_clean) > 200 else ''}")
    else:
        steps.append(f"4. Limpieza de caracteres: No se encontraron caracteres especiales para eliminar")
    
    # Paso 5: Tokenización
    tokens = word_tokenize(text_clean)
    steps.append(f"5. Tokenización completada: {len(tokens)} tokens extraídos")
    steps.append(f"   Primeros tokens: {' | '.join(tokens[:15])}{'...' if len(tokens) > 15 else ''}")
    
    # Paso 6: Análisis de stopwords
    tokens_stopwords = [token for token in tokens if token in stop_words]
    tokens_kept_stopwords = [token for token in tokens if token in keep_words]
    
    if tokens_stopwords:
        unique_stopwords = list(set(tokens_stopwords))
        steps.append(f"6. Eliminacion de Stopword: ({len(tokens_stopwords)} total, {len(unique_stopwords)} únicas)")
        steps.append(f"   Stopwords encontradas: {', '.join(unique_stopwords[:20])}{'...' if len(unique_stopwords) > 20 else ''}")
        if tokens_kept_stopwords:
            steps.append(f"   Stopwords preservadas (negaciones/intensificadores): {', '.join(set(tokens_kept_stopwords))}")
    else:
        steps.append(f"6. Stopwords: No se encontraron stopwords en el texto")
    
    # Paso 7: Filtrado efectivo de stopwords
    tokens_no_stop = [token for token in tokens if token not in stop_words]
    removed_count = len(tokens) - len(tokens_no_stop)
    
    if removed_count > 0:
        steps.append(f"Filtrado de stopwords: {removed_count} palabras eliminadas ({len(tokens)} → {len(tokens_no_stop)} tokens)")
        steps.append(f"Tokens tras filtrado: {' | '.join(tokens_no_stop[:15])}{'...' if len(tokens_no_stop) > 15 else ''}")
    else:
        steps.append(f"Filtrado de stopwords: No se eliminaron tokens (no había stopwords)")
    
    # Paso 8: Lematización con seguimiento de cambios
    tokens_lemmatized = []
    lemmatization_changes = []
    
    for token in tokens_no_stop:
        lemmatized = lemmatizer.lemmatize(token)
        tokens_lemmatized.append(lemmatized)
        if lemmatized != token:
            lemmatization_changes.append(f"{token}→{lemmatized}")
    
    if lemmatization_changes:
        steps.append(f"7. Lematización aplicada: {len(lemmatization_changes)} palabras modificadas")
        steps.append(f"   Cambios principales: {', '.join(lemmatization_changes[:12])}{'...' if len(lemmatization_changes) > 12 else ''}")
    else:
        steps.append(f"7. Lematización: No se requirieron cambios morfológicos")
    
    # Paso 9: Resultado final y estadísticas
    final_text = ' '.join(tokens_lemmatized)
    steps.append(f"Texto procesado final ({len(tokens_lemmatized)} tokens): {final_text[:250]}{'...' if len(final_text) > 250 else ''}")
    
    # Resumen estadístico de la compresión
    compression_ratio = (len(final_text) / len(text)) * 100 if len(text) > 0 else 0
    steps.append(f"Resumen: {len(text)} → {len(final_text)} caracteres ({compression_ratio:.1f}% del original)")
    
    return steps

# =============================================================================
# ARQUITECTURA DEL MODELO BERT PARA NOTICIAS
# =============================================================================
class BERTClassifier(nn.Module):
    """
    Clasificador personalizado basado en DistilBERT para detección de noticias falsas.
    
    Arquitectura:
    - Base: DistilBERT pre-entrenado
    - Dropout para regularización
    - Capa lineal final para clasificación binaria
    """
    
    def __init__(self, n_classes=2, dropout_rate=0.3):
        super(BERTClassifier, self).__init__()
        
        # Cargar DistilBERT pre-entrenado (versión más ligera de BERT)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Capa de dropout para prevenir overfitting
        self.dropout = nn.Dropout(dropout_rate)
        
        # Capa de clasificación final
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Paso hacia adelante del modelo.
        
        Args:
            input_ids: IDs de tokens de entrada
            attention_mask: Máscara de atención
            
        Returns:
            Logits de clasificación
        """
        # Obtener representaciones de BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Usar el token [CLS] para clasificación
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Aplicar dropout y clasificación
        output = self.dropout(pooled_output)
        return self.classifier(output)

# =============================================================================
# ARQUITECTURA DEL MODELO MULTIMODAL (BERT + RESNET)
# =============================================================================
class BERTResNetClassifier(nn.Module):
    """
    Modelo multimodal que combina análisis de texto (BERT) e imagen (ResNet)
    para detectar contenido falso en redes sociales.
    
    Arquitectura:
    - Rama de texto: BERT para embeddings semánticos
    - Rama de imagen: ResNet50 para características visuales
    - Fusión: Operación MAX entre características de texto e imagen
    """
    
    def __init__(self, num_classes=6):
        super(BERTResNetClassifier, self).__init__()
        self.num_classes = num_classes

        # Procesamiento de imagen con ResNet50 pre-entrenado
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Capa totalmente conectada para características de imagen
        self.fc_image = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        
        # Capa de dropout compartida
        self.drop = nn.Dropout(p=0.3)
        
        # Procesamiento de texto con BERT completo
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        
        # Capa totalmente conectada para características de texto
        self.fc_text = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=num_classes, bias=True)
        
        # Función de activación para salida final
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, text_input_ids, text_attention_mask):
        """
        Paso hacia adelante del modelo multimodal.
        
        Args:
            image: Tensor de imagen preprocesada
            text_input_ids: IDs de tokens del texto
            text_attention_mask: Máscara de atención del texto
            
        Returns:
            Logits fusionados de clasificación multimodal
        """
        # Rama de imagen: ResNet50 → Dropout → FC
        x_img = self.image_model(image)
        x_img = self.drop(x_img)
        x_img = self.fc_image(x_img)

        # Rama de texto: BERT → [CLS] token → Dropout → FC
        x_text_last_hidden_states = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=False
        )
        
        # Extraer representación del token [CLS]
        x_text_pooled_output = x_text_last_hidden_states[0][:, 0, :]
        x_text = self.drop(x_text_pooled_output)
        x_text = self.fc_text(x_text_pooled_output)

        # Fusión multimodal usando operación MAX element-wise
        x = torch.max(x_text, x_img)

        return x

# =============================================================================
# FUNCIÓN PARA EMBEDDINGS BERT DEL MODELO MULTIMODAL
# =============================================================================
def get_bert_embedding_multimodal(text, tokenizer):
    """
    Genera embeddings BERT específicamente para el modelo multimodal.
    
    Args:
        text: Texto a procesar
        tokenizer: Tokenizer de BERT
        
    Returns:
        tuple: (input_ids, attention_mask) procesados
    """
    inputs = tokenizer.encode_plus(
        text, 
        add_special_tokens=True,     # Agregar [CLS] y [SEP]
        return_tensors='pt',         # Retornar tensores de PyTorch
        max_length=80,               # Longitud máxima para redes sociales
        truncation=True,             # Truncar si es necesario
        padding='max_length'         # Rellenar hasta longitud máxima
    )
    return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

# =============================================================================
# FUNCIÓN DE PREPROCESAMIENTO DE IMÁGENES
# =============================================================================
def preprocess_image_for_multimodal(image_bytes):
    """
    Preprocesa imágenes para el modelo multimodal aplicando las mismas
    transformaciones usadas durante el entrenamiento.
    
    Args:
        image_bytes: Bytes de la imagen cargada
        
    Returns:
        Tensor de imagen preprocesada listo para el modelo
    """
    # Parámetros de normalización de ImageNet (estándar para ResNet)
    img_size = 256
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # Pipeline de transformaciones
    transform_func = v2.Compose([
        v2.Resize(256),                          # Redimensionar a 256x256
        v2.ToImage(),                            # Convertir a tensor imagen
        v2.ToDtype(torch.float32, scale=True),   # Normalizar a [0,1]
        v2.Normalize(mean, std)                  # Normalización ImageNet
    ])
    
    # Cargar imagen desde bytes y convertir a RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Aplicar transformaciones
    processed_image = transform_func(image)
    
    # Agregar dimensión de batch [1, C, H, W]
    return processed_image.unsqueeze(0)

# =============================================================================
# FUNCIÓN DE PREDICCIÓN DE CATEGORÍAS CON K-MEANS
# =============================================================================
def predict_category_with_kmeans(text, kmeans_model, kmeans_vectorizer):
    """
    Predice la categoría temática del texto usando clustering K-means.
    
    Args:
        text: Texto a clasificar
        kmeans_model: Modelo K-means entrenado
        kmeans_vectorizer: Vectorizador TF-IDF correspondiente
        
    Returns:
        tuple: (nombre_categoria, términos_representativos, error_si_existe)
    """
    if kmeans_model is None or kmeans_vectorizer is None:
        print("Modelos K-means no disponibles para predicción de categoría")
        return "desconocida", [], "Modelos K-means no cargados"
    
    try:
        # Preprocesar el texto usando la misma función que en entrenamiento
        processed_text = preprocess_text(text)
        if not processed_text.strip():
            return "desconocida", [], "Texto vacío después del preprocesamiento"
        
        # Vectorizar usando TF-IDF entrenado
        text_vector = kmeans_vectorizer.transform([processed_text])
        
        # Predecir cluster más cercano
        cluster = kmeans_model.predict(text_vector)[0]
        
        # Obtener información semántica del cluster
        category_name, cluster_terms = get_cluster_terms(cluster)
        
        print(f"Categoría predicha: {category_name} (cluster {cluster})")
        print(f"Términos del cluster: {cluster_terms}")
        
        return category_name, cluster_terms, None
        
    except Exception as e:
        error_msg = f"Error en predicción de categoría: {str(e)}"
        print(f"Error: {error_msg}")
        return "desconocida", [], error_msg

# =============================================================================
# FUNCIÓN PARA CREAR DATAFRAMES SEGÚN TIPO DE ANÁLISIS
# =============================================================================
def create_dataframe_for_analysis(title, body, analysis_type, predicted_category):
    """
    Crea DataFrames con la estructura correcta según el tipo de análisis
    que se va a realizar (título, cuerpo o combinado).
    
    Args:
        title: Título de la noticia
        body: Cuerpo de la noticia
        analysis_type: Tipo de análisis ('title', 'body', 'combined')
        predicted_category: Categoría predicha por K-means
        
    Returns:
        DataFrame con la estructura apropiada para los modelos
    """
    title = str(title) if title else ""
    body = str(body) if body else ""
    
    if analysis_type == 'title':
        # Análisis de título: columna 'title' requerida
        data_dict = {
            'title': title,
            'category': str(predicted_category)
        }
        
    elif analysis_type == 'body':
        # Análisis de cuerpo: columna 'text' requerida
        data_dict = {
            'text': body,
            'category': str(predicted_category)
        }
        
    elif analysis_type == 'combined':
        # Análisis combinado: columna 'sentences' requerida
        sentences_text = f"{title} {body}".strip() if title and body else (title or body)
        data_dict = {
            'sentences': sentences_text,
            'category': str(predicted_category)
        }
        
    else:
        raise ValueError(f"Tipo de análisis no reconocido: {analysis_type}")
    
    # Crear DataFrame con estructura apropiada
    input_data = pd.DataFrame([data_dict])
    
    return input_data

# =============================================================================
# FUNCIÓN DE CARGA DE TODOS LOS MODELOS
# =============================================================================
def load_models():
    """
    Carga todos los modelos necesarios para la aplicación con manejo robusto
    de errores. Incluye modelos tradicionales, BERT y multimodal.
    
    Returns:
        tuple: (modelos, tokenizer, device, kmeans_model, kmeans_vectorizer)
    """
    global model_loading_errors
    model_loading_errors = []
    
    # Estructura para organizar todos los modelos
    models = {
        'news': {
            'title': {},           # Modelos para análisis de títulos
            'body': {},            # Modelos para análisis de cuerpo
            'combined': {},        # Modelos para análisis combinado
            'bert_models': {}      # Modelos BERT especializados
        },
        'social_media': {
            'multimodal': None,    # Modelo multimodal
            'tokenizer': None      # Tokenizer específico
        }
    }
    
    tokenizer = None
    # Detectar si hay GPU disponible para acelerar inferencia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kmeans_model = None
    kmeans_vectorizer = None
    
    print("Iniciando carga de modelos...")
    
    # Cargar tokenizer BERT para noticias
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("Tokenizer BERT cargado exitosamente")
    except Exception as e:
        error = f"Error cargando tokenizer BERT: {str(e)}"
        print(f"Error: {error}")
        model_loading_errors.append(error)
    
    # Cargar tokenizer para modelo multimodal
    try:
        multimodal_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        models['social_media']['tokenizer'] = multimodal_tokenizer
        print("Tokenizer multimodal cargado exitosamente")
    except Exception as e:
        error = f"Error cargando tokenizer multimodal: {str(e)}"
        print(f"Error: {error}")
        model_loading_errors.append(error)
    
    # Cargar modelo multimodal para redes sociales
    try:
        multimodal_path = Path('models/multimodal.pth')
        if multimodal_path.exists():
            multimodal_model = BERTResNetClassifier(num_classes=6)
            multimodal_model.load_state_dict(torch.load(multimodal_path, map_location=device))
            multimodal_model.to(device)
            multimodal_model.eval()  # Modo evaluación (sin dropout)
            models['social_media']['multimodal'] = multimodal_model
            print("Modelo multimodal cargado exitosamente")
        else:
            error = f"Archivo no encontrado: {multimodal_path}"
            print(f"Error: {error}")
            model_loading_errors.append(error)
    except Exception as e:
        error = f"Error cargando modelo multimodal: {str(e)}\n{traceback.format_exc()}"
        print(f"Error: {error}")
        model_loading_errors.append(error)
    
    # Cargar modelo K-means para categorización
    try:
        kmeans_path = Path('models/kmeans_model.pkl')
        if kmeans_path.exists():
            kmeans_model = joblib.load(kmeans_path)
            print("Modelo K-means cargado exitosamente")
        else:
            error = f"Archivo no encontrado: {kmeans_path}"
            print(f"Error: {error}")
            model_loading_errors.append(error)
    except Exception as e:
        error = f"Error cargando K-means: {str(e)}\n{traceback.format_exc()}"
        print(f"Error: {error}")
        model_loading_errors.append(error)
    
    # Cargar vectorizador TF-IDF para K-means
    try:
        vectorizer_path = Path('models/kmeans_vectorizer.pkl')
        if vectorizer_path.exists():
            kmeans_vectorizer = joblib.load(vectorizer_path)
            print("Vectorizer K-means cargado exitosamente")
        else:
            error = f"Archivo no encontrado: {vectorizer_path}"
            print(f"Error: {error}")
            model_loading_errors.append(error)
    except Exception as e:
        error = f"Error cargando vectorizer K-means: {str(e)}\n{traceback.format_exc()}"
        print(f"Error: {error}")
        model_loading_errors.append(error)
    
    # Definir rutas de modelos tradicionales de machine learning
    model_paths = {
        'title': {
            'Naive Bayes': 'models/naive_bayes_title.pkl',
            'SVM': 'models/SVM_title.pkl',
            'Random Forest': 'models/random_forest_title.pkl'
        },
        'body': {
            'Naive Bayes': 'models/naive_bayes_body.pkl',
            'SVM': 'models/SVM_body.pkl', 
            'Random Forest': 'models/random_forest_body.pkl'
        },
        'combined': {
            'Naive Bayes': 'models/naive_bayes_total.pkl',
            'SVM': 'models/SVM_total.pkl',
            'Random Forest': 'models/random_forest_total.pkl'
        }
    }
    
    # Cargar todos los modelos tradicionales
    for analysis_type in ['title', 'body', 'combined']:
        for model_name, path in model_paths[analysis_type].items():
            try:
                model_path = Path(path)
                if model_path.exists():
                    models['news'][analysis_type][model_name] = joblib.load(model_path)
                    print(f"Cargado {model_name} para {analysis_type}")
                else:
                    error = f"Archivo no encontrado: {path}"
                    print(f"Error: {error}")
                    model_loading_errors.append(error)
            except Exception as e:
                error = f"Error cargando {model_name} para {analysis_type}: {str(e)}\n{traceback.format_exc()}"
                print(f"Error: {error}")
                model_loading_errors.append(error)
    
    # Definir rutas de modelos BERT especializados
    bert_paths = {
        'title': 'models/bert_title.pth',
        'body': 'models/bert_text.pth', 
        'combined': 'models/bert_sentences.pth'
    }
    
    # Cargar modelos BERT para cada tipo de análisis
    for analysis_type, path in bert_paths.items():
        try:
            bert_path = Path(path)
            if bert_path.exists():
                model = BERTClassifier()
                model.load_state_dict(torch.load(bert_path, map_location=device))
                model.to(device)
                model.eval()  # Modo evaluación
                models['news']['bert_models'][analysis_type] = model
                print(f"Cargado BERT para {analysis_type}")
            else:
                error = f"Archivo no encontrado: {path}"
                print(f"Error: {error}")
                model_loading_errors.append(error)
        except Exception as e:
            error = f"Error cargando BERT para {analysis_type}: {str(e)}\n{traceback.format_exc()}"
            print(f"Error: {error}")
            model_loading_errors.append(error)
    
    print(f"Carga de modelos completada. Errores encontrados: {len(model_loading_errors)}")
    return models, tokenizer, device, kmeans_model, kmeans_vectorizer

# =============================================================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN PARA NOTICIAS
# =============================================================================
def make_real_prediction(title, body, analysis_type, models, tokenizer, device, kmeans_model=None, kmeans_vectorizer=None):
    """
    Ejecuta predicciones con todos los modelos disponibles para noticias.
    Combina modelos tradicionales de ML y BERT para análisis robusto.
    
    Args:
        title: Título de la noticia
        body: Cuerpo de la noticia
        analysis_type: Tipo de análisis ('title', 'body', 'combined')
        models: Diccionario con todos los modelos cargados
        tokenizer: Tokenizer de BERT
        device: Dispositivo (CPU/GPU)
        kmeans_model: Modelo K-means para categorización
        kmeans_vectorizer: Vectorizador TF-IDF
        
    Returns:
        tuple: (resultados, errores, categoría, términos_cluster)
    """
    # Determinar texto principal según tipo de análisis
    if analysis_type == 'title':
        main_text = title
    elif analysis_type == 'body':
        main_text = body
    elif analysis_type == 'combined':
        main_text = f"{title} {body}".strip()
    else:
        return {}, f"Tipo de análisis no reconocido: {analysis_type}", "desconocida", []
    
    if not main_text.strip():
        return {}, "Texto vacío", "desconocida", []
    
    results = {}
    prediction_errors = []
    
    # Predecir categoría temática usando K-means
    predicted_category, cluster_terms, category_error = predict_category_with_kmeans(main_text, kmeans_model, kmeans_vectorizer)
    if category_error:
        prediction_errors.append(f"Categoría: {category_error}")
        predicted_category = "general"
        cluster_terms = []
    
    # Crear DataFrame con estructura apropiada para los modelos
    try:
        input_data = create_dataframe_for_analysis(title, body, analysis_type, predicted_category)
        print(f"Análisis tipo: {analysis_type}")
        print(f"Estructura DataFrame:")
        print(f"   Columnas: {list(input_data.columns)}")
        print(f"   Tipos: {input_data.dtypes.to_dict()}")
        print(f"   Valores: {input_data.iloc[0].to_dict()}")
        
    except Exception as e:
        error_msg = f"Error creando DataFrame: {str(e)}"
        print(f"Error: {error_msg}")
        return {}, error_msg, predicted_category, cluster_terms
    
    # Ejecutar predicciones con modelos tradicionales
    traditional_models = models['news'].get(analysis_type, {})
    if not traditional_models:
        prediction_errors.append(f"No hay modelos tradicionales cargados para {analysis_type}")
    
    # Iterar sobre cada modelo tradicional disponible
    for model_name, model_pipeline in traditional_models.items():
        if model_pipeline is not None:
            try:
                print(f"\nProbando {model_name}...")
                
                # Preparar datos para el modelo
                model_input = input_data.copy()
                features_used = model_input.iloc[0].to_dict()
                
                # Realizar predicción
                prediction = model_pipeline.predict(model_input)[0]
                
                # Calcular confianza (probabilidad o score de decisión)
                confidence = 0.5  # Valor por defecto
                if hasattr(model_pipeline, 'predict_proba'):
                    try:
                        probabilities = model_pipeline.predict_proba(model_input)[0]
                        confidence = float(max(probabilities))
                    except Exception as prob_error:
                        print(f"   Warning predict_proba: {prob_error}")
                        
                        # Fallback a decision_function para SVM
                        if hasattr(model_pipeline, 'decision_function'):
                            try:
                                decision_score = model_pipeline.decision_function(model_input)[0]
                                confidence = min(0.5 + abs(float(decision_score)) / 10, 1.0)
                            except Exception as dec_error:
                                print(f"   Warning decision_function: {dec_error}")
                
                # Almacenar resultados
                results[model_name] = {
                    'prediction': 'FALSA' if prediction == 1 else 'VERDADERA',
                    'confidence': confidence,
                    'is_fake': bool(prediction == 1),
                    'category': str(predicted_category),
                    'features': features_used  # Para explicabilidad
                }
                print(f"   Éxito - Predicción: {results[model_name]['prediction']}")
                
            except Exception as e:
                error_msg = f"Error en {model_name}: {str(e)}"
                print(f"   Error: {error_msg}")
                prediction_errors.append(error_msg)
    
    # Ejecutar predicción con modelo BERT
    bert_model = models['news'].get('bert_models', {}).get(analysis_type)
    if bert_model is not None and tokenizer is not None:
        try:
            print(f"\nProbando BERT...")
            
            # Tokenizar texto para BERT
            encoding = tokenizer(
                main_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Mover tensores al dispositivo apropiado
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Realizar inferencia sin calcular gradientes
            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            # Almacenar resultados de BERT
            results['BERT'] = {
                'prediction': 'FALSA' if prediction == 1 else 'VERDADERA',
                'confidence': confidence,
                'is_fake': prediction == 1,
                'category': predicted_category
            }
            print(f"   BERT - Predicción: {results['BERT']['prediction']}")
            
        except Exception as e:
            error_msg = f"Error en BERT: {str(e)}"
            print(f"   Error: {error_msg}")
            prediction_errors.append(error_msg)
    else:
        if bert_model is None:
            prediction_errors.append(f"Modelo BERT no disponible para {analysis_type}")
        if tokenizer is None:
            prediction_errors.append("Tokenizer BERT no disponible")
    
    # Compilar errores si los hay
    error_summary = "; ".join(prediction_errors) if prediction_errors else None
    print("FEATURES GUARDADAS:", input_data.iloc[0].to_dict() if len(results) > 0 else "No hay resultados")
    
    return results, error_summary, predicted_category, cluster_terms

# =============================================================================
# FUNCIÓN DE PREDICCIÓN PARA REDES SOCIALES (MULTIMODAL)
# =============================================================================
def make_social_media_prediction(text, image_bytes, models, device):
    """
    Ejecuta predicción multimodal para contenido de redes sociales,
    combinando análisis de texto e imagen cuando está disponible.
    
    Args:
        text: Contenido textual del post
        image_bytes: Bytes de la imagen (puede ser None)
        models: Diccionario con modelos cargados
        device: Dispositivo de cómputo
        
    Returns:
        tuple: (resultados, errores, None, []) - None y [] porque no usa K-means
    """
    if not text.strip():
        return {}, "Texto vacío", None, []
    
    results = {}
    prediction_errors = []
    
    # Obtener modelo y tokenizer multimodal
    multimodal_model = models['social_media'].get('multimodal')
    multimodal_tokenizer = models['social_media'].get('tokenizer')
    
    if multimodal_model is not None and multimodal_tokenizer is not None:
        try:
            print("Realizando predicción multimodal...")
            
            # Procesar texto con BERT tokenizer
            input_ids, attention_mask = get_bert_embedding_multimodal(text, multimodal_tokenizer)
            input_ids = input_ids.unsqueeze(0).to(device)      # [1, seq_len]
            attention_mask = attention_mask.unsqueeze(0).to(device)  # [1, seq_len]
            
            # Procesar imagen si está disponible
            if image_bytes is not None:
                try:
                    processed_image = preprocess_image_for_multimodal(image_bytes)
                    processed_image = processed_image.to(device)
                    analysis_type = "Multimodal (Texto + Imagen)"
                except Exception as img_error:
                    print(f"Error procesando imagen: {img_error}")
                    # Usar imagen en negro como fallback
                    processed_image = torch.zeros((1, 3, 256, 256)).to(device)
                    analysis_type = "Solo Texto (error en imagen)"
                    prediction_errors.append(f"Error en imagen: {img_error}")
            else:
                # Usar imagen en negro para análisis solo de texto
                processed_image = torch.zeros((1, 3, 256, 256)).to(device)
                analysis_type = "Solo Texto"
            
            # Realizar predicción multimodal
            with torch.no_grad():
                outputs = multimodal_model(processed_image, input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            # Mapear predicción numérica a etiqueta descriptiva
            prediction_label = multimodal_labels.get(prediction, f"CLASE_{prediction}")
            is_fake = prediction != 0  # 0 = TRUE, resto = tipos de falsedad
            
            # Almacenar resultados multimodales
            results['Modelo Multimodal'] = {
                'prediction': prediction_label,
                'confidence': confidence,
                'is_fake': is_fake
            }
            
            print(f"Predicción multimodal: {prediction_label}")
            
        except Exception as e:
            error_msg = f"Error en modelo multimodal: {str(e)}"
            print(f"Error: {error_msg}")
            prediction_errors.append(error_msg)
            
    else:
        if multimodal_model is None:
            prediction_errors.append("Modelo multimodal no disponible")
        if multimodal_tokenizer is None:
            prediction_errors.append("Tokenizer multimodal no disponible")
    
    error_summary = "; ".join(prediction_errors) if prediction_errors else None
    return results, error_summary, None, []

# =============================================================================
# INICIALIZACIÓN DE MODELOS AL ARRANQUE
# =============================================================================
# Cargar todos los modelos al inicializar la aplicación
try:
    models, tokenizer, device, kmeans_model, kmeans_vectorizer = load_models()
    print("Sistema de modelos inicializado exitosamente")
except Exception as e:
    print(f"Error crítico cargando modelos: {e}")
    print(traceback.format_exc())
    
    # Inicializar con estructura vacía en caso de error
    models, tokenizer, device, kmeans_model, kmeans_vectorizer = {
        'news': {
            'title': {}, 
            'body': {}, 
            'combined': {}, 
            'bert_models': {}
        }, 
        'social_media': {
            'multimodal': None,
            'tokenizer': None
        }
    }, None, None, None, None

# =============================================================================
# ESTILOS CSS 
# =============================================================================
custom_css = """
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">

<style>
/* =============================================================================
   VARIABLES CSS PARA COLORES Y EFECTOS CONSISTENTES
   ============================================================================= */
:root {
    /* Paleta de colores principal */
    --violet-600: #1e3a8a;
    --violet-700: #1e40af;
    --violet-500: #3b82f6;
    --indigo-600: #1e40af;
    --cyan-500: #0ea5e9;
    --emerald-500: #059669;
    --orange-500: #d97706;
    --red-500: #dc2626;
    --pink-500: #e11d48;
    
    /* Escala de grises moderna */
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-400: #9ca3af;
    --gray-500: #6b7280;
    --gray-600: #4b5563;
    --gray-700: #374151;
    --gray-800: #1f2937;
    --gray-900: #111827;
    
    /* Efectos de sombra sin brillos */
    --shadow-glow: 0 4px 12px rgba(30, 58, 138, 0.15);
    --shadow-card: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    
    /* Fondos con transparencia */
    --glass-bg: rgba(255, 255, 255, 0.95);
    --glass-border: rgba(203, 213, 225, 0.5);
}

/* Reset CSS básico */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Estilos base del body */
body {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--gray-50);
    min-height: 100vh;
    color: var(--gray-800);
    line-height: 1.6;
}

/* =============================================================================
   SECCIÓN HERO CON ANIMACIONES
   ============================================================================= */
.hero-section {
    background-color: var(--violet-600);
    padding: 4rem 0;
    position: relative;
}

/* Partículas animadas de fondo */
.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="40" r="1.5" fill="rgba(255,255,255,0.1)"/><circle cx="40" cy="70" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="70" cy="80" r="2.5" fill="rgba(255,255,255,0.1)"/></svg>');
    animation: float 20s linear infinite;
}

@keyframes float {
    0% { transform: translateY(0px) rotate(0deg); }
    100% { transform: translateY(-100px) rotate(360deg); }
}

.hero-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
    text-align: center;
    position: relative;
    z-index: 1;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.hero-subtitle {
    font-size: 1.25rem;
    color: rgba(255, 255, 255, 0.9);
    max-width: 600px;
    margin: 0 auto;
    font-weight: 400;
}

/* =============================================================================
   CONTENEDOR PRINCIPAL Y NAVEGACIÓN
   ============================================================================= */
.app-container {
    max-width: 1400px;
    margin: -3rem auto 0;
    padding: 0 2rem 4rem;
    position: relative;
    z-index: 10;
}

.nav-container {
    display: flex;
    justify-content: center;
    margin-bottom: 3rem;
}

.nav-tabs {
    display: flex;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 2rem;
    padding: 0.5rem;
    box-shadow: var(--shadow-glow);
}

.nav-tab {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem 2rem;
    background: transparent;
    border: none;
    border-radius: 1.5rem;
    color: rgba(75, 85, 99, 0.8);
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
}

.nav-tab:hover {
    background: var(--gray-100);
    color: var(--gray-800);
    transform: translateY(-1px);
}

.nav-tab.active {
    background-color: var(--violet-600);
    color: white;
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4);
    transform: translateY(-1px);
}

/* =============================================================================
   PANELES Y CARDS DE CONTENIDO
   ============================================================================= */
.content-panel {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: 1.5rem;
    box-shadow: var(--shadow-lg);
    margin-bottom: 2rem;
    overflow: hidden;
}

.panel-header {
    background: linear-gradient(135deg, var(--violet-600), var(--indigo-600));
    padding: 2rem;
    border-bottom: none;
}

.panel-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: white;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.panel-title i {
    color: white;
    font-size: 1.5rem;
}

.panel-body {
    padding: 2.5rem;
}

/* =============================================================================
   SISTEMA DE FORMULARIOS
   ============================================================================= */
.form-layout {
    display: grid;
    gap: 2rem;
}

.form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

.input-group {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.input-label {
    font-weight: 600;
    color: var(--gray-700);
    font-size: 0.95rem;
}

.form-control {
    padding: 1rem 1.25rem;
    border: 2px solid var(--gray-200);
    border-radius: 0.75rem;
    font-size: 1rem;
    transition: all 0.3s ease;
    background: white;
    font-family: inherit;
}

.form-control:focus {
    outline: none;
    border-color: var(--violet-500);
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.1);
}

.form-textarea {
    min-height: 140px;
    resize: vertical;
}

/* =============================================================================
   BOTONES CON EFECTOS MODERNOS
   ============================================================================= */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 1rem 2rem;
    border: none;
    border-radius: 0.75rem;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-decoration: none;
    position: relative;
    overflow: hidden;
}

/* Efecto de brillo al hacer hover */
.btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.btn:hover::before {
    left: 100%;
}

.btn-primary {
    background-color: var(--violet-600);
    color: white;
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
}

.btn-primary:hover {
    background-color: var(--violet-700);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(30, 58, 138, 0.4);
}

.btn-secondary {
    background-color: var(--gray-100);
    color: var(--gray-700);
    box-shadow: 0 4px 12px rgba(107, 114, 128, 0.2);
}

.btn-secondary:hover {
    background-color: var(--gray-200);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(107, 114, 128, 0.3);
}

.btn-accent {
    background-color: var(--cyan-500);
    color: white;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
}

.btn-accent:hover {
    background-color: #0284c7;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(14, 165, 233, 0.4);
}

.btn-lg {
    padding: 1.25rem 2.5rem;
    font-size: 1.125rem;
    border-radius: 1rem;
}

/* =============================================================================
   ESTILOS PARA INPUT DE ARCHIVOS PERSONALIZADO (SIN BOTÓN NATIVO)
   ============================================================================= */

.input-group {
    position: relative;
    width: 100%;
}

/* Ocultar completamente el input file nativo */
.input-group input[type="file"] {
    display: block;
    width: 100%;
    padding: 1rem;
    border: 2px dashed var(--violet-600);
    border-radius: 1rem;
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.05), rgba(14, 165, 233, 0.05));
    font-weight: 600;
    color: var(--violet-600);
    cursor: pointer;
    transition: all 0.3s ease;
}

/* =============================================================================
   GRILLA DE RESULTADOS Y TARJETAS
   ============================================================================= */
.analysis-controls {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    margin: 2rem 0;
    padding: 2rem;
    background: linear-gradient(135deg, var(--gray-50), var(--gray-100));
    border-radius: 1rem;
    border: 1px solid var(--gray-200);
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
}

.result-card {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem;
    border: 1px solid var(--gray-200);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

/* Barra superior de color en cada tarjeta */
.result-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--violet-500), var(--cyan-500));
}

.result-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.model-name {
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--gray-800);
    margin-bottom: 1rem;
}

/* =============================================================================
   BADGES DE PREDICCIÓN CON COLORES SEMÁNTICOS
   ============================================================================= */
.prediction-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
}

.prediction-real {
    background: linear-gradient(135deg, var(--emerald-500), #22c55e);
    color: white;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
}

.prediction-fake {
    background: linear-gradient(135deg, var(--red-500), #dc2626);
    color: white;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
}

.prediction-other {
    background: linear-gradient(135deg, var(--orange-500), #f97316);
    color: white;
    box-shadow: 0 4px 15px rgba(249, 115, 22, 0.3);
}

.confidence-text {
    color: var(--gray-600);
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* Barras de confianza con estilo monospace */
.confidence-bar {
    margin: 0.75rem 0;
    color: var(--gray-700);
    font-weight: 600;
}

.confidence-bar span {
    color: var(--violet-600);
    display: block;
    margin-top: 0.25rem;
}

/* =============================================================================
   VISTA PREVIA DE IMÁGENES
   ============================================================================= */
.image-preview-section {
    text-align: center;
    margin: 2rem 0;
    padding: 2rem;
    background-color: var(--gray-50);
    border-radius: 1rem;
    border: 2px dashed var(--gray-300);
}

.preview-image {
    max-width: 400px;
    max-height: 300px;
    border-radius: 1rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    border: 3px solid white;
}

/* =============================================================================
   SISTEMA DE ALERTAS MODERNAS
   ============================================================================= */
.alert {
    padding: 1.25rem 1.5rem;
    border-radius: 1rem;
    margin: 1.5rem 0;
    border: 1px solid;
    backdrop-filter: blur(10px);
}

.alert-info {
    background-color: rgba(14, 165, 233, 0.1);
    border-color: rgba(14, 165, 233, 0.3);
    color: var(--indigo-600);
}

.alert-warning {
    background-color: rgba(217, 119, 6, 0.1);
    border-color: rgba(217, 119, 6, 0.3);
    color: var(--orange-500);
}

.alert-success {
    background-color: rgba(5, 150, 105, 0.1);
    border-color: rgba(5, 150, 105, 0.3);
    color: var(--emerald-500);
}

.alert-error {
    background-color: rgba(220, 38, 38, 0.1);
    border-color: rgba(220, 38, 38, 0.3);
    color: var(--red-500);
}

/* =============================================================================
   SECCIÓN DE ERRORES DETALLADOS
   ============================================================================= */
.error-details {
    background: rgba(220, 38, 38, 0.05);
    border: 1px solid rgba(220, 38, 38, 0.2);
    border-radius: 1rem;
    padding: 1.5rem;
    margin: 1rem 0;
    font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    max-height: 300px;
    overflow-y: auto;
}

.error-title {
    color: var(--red-500);
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* =============================================================================
   INFORMACIÓN DE CATEGORÍAS TEMÁTICAS
   ============================================================================= */
.category-info {
    background: linear-gradient(135deg, var(--violet-600), var(--indigo-600));
    color: white;
    padding: 2rem;
    border-radius: 1.5rem;
    margin: 1.5rem 0;
    text-align: center;
    box-shadow: 0 8px 24px rgba(30, 58, 138, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.category-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    color: white;
}

.category-terms {
    background: rgba(255, 255, 255, 0.15);
    padding: 1rem;
    border-radius: 0.75rem;
    margin-top: 1rem;
    font-size: 0.95rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.category-terms strong {
    color: rgba(255, 255, 255, 0.9);
    display: block;
    margin-bottom: 0.5rem;
}

/* =============================================================================
   SECCIÓN DE PREPROCESAMIENTO (RESTAURADA)
   ============================================================================= */
.preprocessing-panel {
    background: white;
    color: var(--gray-800);
    border-radius: 1.5rem;
    margin-top: 2rem;
    border: 1px solid var(--gray-200);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
}

.preprocessing-header {
    background: linear-gradient(135deg, var(--violet-600), var(--indigo-600));
    padding: 2rem;
    border-bottom: none;
}

.preprocessing-title {
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.preprocessing-title i {
    color: white;
    font-size: 1.25rem;
}

.preprocessing-content {
    padding: 2.5rem;
}

.preprocessing-steps {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

/* Estilo solo para el label que actúa como botón */
.custom-upload-btn {
  display: inline-block;
  padding: 1rem;
  border: 2px dashed var(--violet-600);
  border-radius: 1rem;
  background: linear-gradient(135deg, rgba(30,58,138,0.05), rgba(14,165,233,0.05));
  color: var(--violet-600);
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}
.custom-upload-btn:hover {
  background: linear-gradient(135deg, rgba(30,58,138,0.15), rgba(14,165,233,0.15));
}


.preprocessing-step {
    background: var(--gray-50);
    padding: 1.5rem;
    border-radius: 1rem;
    border-left: 4px solid var(--violet-600);
    font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
    font-size: 0.9rem;
    line-height: 1.7;
    margin-bottom: 0.5rem;
    word-wrap: break-word;
    overflow-wrap: break-word;
    box-shadow: 0 2px 8px rgba(30, 58, 138, 0.08);
    transition: all 0.3s ease;
    position: relative;
    border: 1px solid var(--gray-200);
}

.preprocessing-step:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(30, 58, 138, 0.12);
    background: white;
}

/* Destacar el paso final del preprocesamiento */
.preprocessing-step:last-child {
    border-left-color: var(--violet-600);
    background: linear-gradient(135deg, #faf5ff, #f3e8ff);
    font-weight: 600;
    border-left-width: 6px;
}

/* =============================================================================
   RESPONSIVE DESIGN
   ============================================================================= */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }
    
    .app-container {
        padding: 0 1rem 2rem;
    }
    
    .form-row {
        grid-template-columns: 1fr;
    }
    
    .nav-tabs {
        flex-direction: column;
        width: 100%;
    }
    
    .analysis-controls {
        flex-direction: column;
        align-items: center;
    }
    
    .results-grid {
        grid-template-columns: 1fr;
    }
}

/* =============================================================================
   UTILIDADES Y ANIMACIONES
   ============================================================================= */
.text-center { text-align: center; }
.mb-4 { margin-bottom: 2rem; }
.mt-4 { margin-top: 2rem; }

.fade-in {
    animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Layouts responsivos para la grilla de resultados */
.results-grid:has(.result-card:nth-child(2):not(.result-card:nth-child(5))) {
    grid-template-columns: repeat(2, 1fr);
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

.results-grid:has(.result-card:first-child:last-child) {
    grid-template-columns: 1fr;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
}

.results-grid:has(.result-card:nth-child(3):last-child) {
    grid-template-columns: repeat(3, 1fr);
    max-width: 1000px;
    margin-left: auto;
    margin-right: auto;
}

</style>

<script>
/* =============================================================================
   JAVASCRIPT PARA NAVEGACIÓN DINÁMICA ENTRE TABS
   ============================================================================= */
document.addEventListener('DOMContentLoaded', function() {
    
    // Función para actualizar el estado activo de los tabs
    function updateActiveTab(activeTabId) {
        // Remover clase active de todos los tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Agregar clase active al tab seleccionado
        const activeTab = document.getElementById(activeTabId);
        if (activeTab) {
            activeTab.classList.add('active');
        }
    }
    
    // Event listeners para los botones de navegación
    const newsTab = document.getElementById('tab_news');
    const socialTab = document.getElementById('tab_social');
    
    // Configurar listener para tab de noticias
    if (newsTab) {
        newsTab.addEventListener('click', function() {
            updateActiveTab('tab_news');
        });
    }
    
    // Configurar listener para tab de redes sociales
    if (socialTab) {
        socialTab.addEventListener('click', function() {
            updateActiveTab('tab_social');
        });
    }
    
    // Observer para detectar cambios dinámicos en el DOM
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                // Reconfigurar listeners cuando el DOM cambie
                const currentNewsTab = document.getElementById('tab_news');
                const currentSocialTab = document.getElementById('tab_social');
                
                // Agregar listener solo si no existe ya
                if (currentNewsTab && !currentNewsTab.hasAttribute('data-listener')) {
                    currentNewsTab.setAttribute('data-listener', 'true');
                    currentNewsTab.addEventListener('click', function() {
                        updateActiveTab('tab_news');
                    });
                }
                
                if (currentSocialTab && !currentSocialTab.hasAttribute('data-listener')) {
                    currentSocialTab.setAttribute('data-listener', 'true');
                    currentSocialTab.addEventListener('click', function() {
                        updateActiveTab('tab_social');
                    });
                }
            }
        });
    });
    
    // Observar cambios en todo el documento
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
</script>
"""

# =============================================================================
# DEFINICIÓN DE LA INTERFAZ DE USUARIO
# =============================================================================
app_ui = ui.page_fluid(
    # Inyectar CSS personalizado completo
    ui.HTML(custom_css),
    
    # Sección hero principal con título y descripción
    ui.div(
        ui.div(
            ui.h1("Detección de noticias y publicaciones falsas", class_="hero-title"),
            ui.p("Detecta automáticamente noticias falsas y desinformación en medios y redes sociales.", 
                 class_="hero-subtitle"),
            class_="hero-content"
        ),
        class_="hero-section"
    ),
    
    # Contenedor principal de la aplicación
    ui.div(
        # Navegación flotante entre secciones
        ui.div(
            ui.div(
                # Tab para análisis de noticias (activo por defecto)
                ui.input_action_button("tab_news", [
                    ui.HTML('<i class="fas fa-newspaper"></i>'),
                    "Noticias"
                ], class_="nav-tab active"),
                
                # Tab para análisis de redes sociales
                ui.input_action_button("tab_social", [
                    ui.HTML('<i class="fab fa-twitter"></i>'),
                    "Redes sociales"
                ], class_="nav-tab"),
                class_="nav-tabs"
            ),
            class_="nav-container"
        ),
        
        # Contenido que cambia dinámicamente según el tab activo
        ui.output_ui("dynamic_content"),
        
        class_="app-container"
    )
)

# =============================================================================
# LÓGICA DEL SERVIDOR (BACKEND)
# =============================================================================
def server(input, output, session):
    """
    Función principal del servidor que maneja toda la lógica de la aplicación.
    Gestiona la navegación entre tabs y coordina las predicciones.
    """
    
    # Estado reactivo para rastrear el tab activo
    current_tab = reactive.Value("news")
    
    # Manejador para cambio a tab de noticias
    @reactive.Effect
    @reactive.event(input.tab_news)
    def _():
        current_tab.set("news")
    
    # Manejador para cambio a tab de redes sociales
    @reactive.Effect
    @reactive.event(input.tab_social)
    def _():
        current_tab.set("social")
    
    # Renderizador dinámico del contenido según tab activo
    @output
    @render.ui
    def dynamic_content():
        """
        Renderiza la interfaz apropiada según el tab seleccionado.
        """
        if current_tab() == "news":
            return create_news_interface()
        else:
            return create_social_interface()
    
    def create_news_interface():
        """
        Crea la interfaz completa para el análisis de noticias.
        Incluye campos para título y cuerpo del artículo.
        """
        return ui.div(
            # Panel principal con formulario de entrada
            ui.div(
                ui.div(
                    ui.h2([
                        ui.HTML('<i class="fas fa-search"></i>'),
                        "Análisis de noticias"
                    ], class_="panel-title"),
                    class_="panel-header"
                ),
                ui.div(
                    ui.div(
                        # Campo para título de la noticia
                        ui.div(
                            ui.div("Título de la noticia", class_="input-label"),
                            ui.input_text("news_title", "", 
                                        placeholder="Introduce el título o encabezado de la noticia...",
                                        width="100%"),
                            class_="input-group"
                        ),
                        
                        # Campo para cuerpo de la noticia
                        ui.div(
                            ui.div("Contenido de la noticia", class_="input-label"),
                            ui.input_text_area("news_body", "", 
                                             placeholder="Pega aquí el contenido completo del artículo para un análisis exhaustivo...",
                                             height="180px",
                                             width="100%"),
                            class_="input-group"
                        ),
                        
                        # Botón de análisis
                        ui.div(
                            ui.input_action_button("btn_analyze_news", [
                                "Analizar noticia"
                            ], class_="btn btn-primary btn-lg"),
                            class_="text-center mt-4"
                        ),
                        
                        class_="form-layout"
                    ),
                    class_="panel-body"
                ),
                class_="content-panel fade-in"
            ),
            
            # Secciones que aparecen tras el análisis
            ui.output_ui("news_results"),        # Resultados de predicción
            ui.output_ui("news_preprocessing")   # Pasos de preprocesamiento
        )
    
    def create_social_interface():
        """
        Crea la interfaz para análisis de redes sociales.
        Incluye campos para texto e imagen (obligatoria).
        """
        return ui.div(
            # Panel principal para redes sociales
            ui.div(
                ui.div(
                    ui.h2([
                        ui.HTML('<i class="fas fa-hashtag"></i>'),
                        "Análisis de redes sociales"
                    ], class_="panel-title"),
                    class_="panel-header"
                ),
                ui.div(
                    ui.div(
                        # Campo para contenido textual
                        ui.div(
                            ui.div("Contenido de la publicación", class_="input-label"),
                            ui.input_text_area("social_text", "", 
                                             placeholder="Introduce el contenido de la publicación, tweet o post...",
                                             height="140px",
                                             width="100%"),
                            class_="input-group"
                        ),
                        
                        # Campo para subir imagen (obligatorio)
                        ui.div(
                            ui.div("Subir imagen", class_="input-label"),
                            ui.tags.div(
                                ui.tags.input(
                                    id="social_image",
                                    type="file",
                                    accept=".jpg,.jpeg,.png,.gif",
                                    style="display: none;"
                                ),
                                ui.tags.label("Haz clic para seleccionar el archivo", **{"for": "social_image"}, class_="custom-upload-btn")
                            ),
                            class_="input-group"
                        ),

                        # Vista previa de imagen subida
                        ui.output_ui("social_image_preview"),
                        
                        # Botón de análisis multimodal
                        ui.div(
                            ui.input_action_button("btn_analyze_social", [
                                "Analizar publicación"
                            ], class_="btn btn-primary btn-lg"),
                            class_="text-center mt-4"
                        ),
                        
                        class_="form-layout"
                    ),
                    class_="panel-body"
                ),
                class_="content-panel fade-in"
            ),
            
            # Solo resultados (sin preprocesamiento para redes sociales)
            ui.output_ui("social_results")
        )
    
    # ==========================================================================
    # VISTA PREVIA DE IMAGEN PARA REDES SOCIALES
    # ==========================================================================
    @output
    @render.ui
    def social_image_preview():
        """
        Renderiza una vista previa de la imagen subida para redes sociales.
        """
        if input.social_image() is None:
            return ui.div()
        
        try:
            file_info = input.social_image()[0]
            
            # Leer y codificar imagen en base64 para mostrar en HTML
            with open(file_info["datapath"], "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            img_src = f"data:{file_info['type']};base64,{img_data}"
            
            return ui.div(
                ui.h4([
                    ui.HTML('<i class="fas fa-image"></i>'),
                    "Vista previa de imagen"
                ], class_="mb-4"),
                ui.img(src=img_src, class_="preview-image"),
                ui.p(f"📎 {file_info['name']}", style="margin-top: 1rem; color: var(--gray-600);"),
                class_="image-preview-section"
            )
        except Exception as e:
            return ui.div(
                ui.div(f"Error cargando imagen: {str(e)}", class_="alert alert-error")
            )
    
    # ==========================================================================
    # MANEJADOR DE ANÁLISIS DE NOTICIAS
    # ==========================================================================
    @output
    @render.ui
    @reactive.event(input.btn_analyze_news)
    def news_results():
        """
        Ejecuta el análisis completo de noticias y renderiza los resultados.
        Maneja análisis de título, cuerpo o combinado según el contenido disponible.
        """
        title = input.news_title()
        body = input.news_body()
        
        # Validar que hay contenido para analizar
        if not title.strip() and not body.strip():
            return ui.div(
                ui.div(
                    ui.HTML('<i class="fas fa-exclamation-triangle"></i> Por favor, introduce al menos un título o contenido del artículo para analizar.'),
                    class_="alert alert-warning"
                )
            )
        
        result_sections = []
        
        # Determinar tipo de análisis y ejecutar predicciones
        if title.strip() and body.strip():
            # =====================================================================
            # ANÁLISIS COMBINADO (TÍTULO + CUERPO)
            # =====================================================================
            results, errors, category, cluster_terms = make_real_prediction(
                title, body, 'combined', models, tokenizer, device, 
                kmeans_model, kmeans_vectorizer
            )
            
            # Mostrar categoría temática detectada
            category_section = ui.div(
                ui.div(
                    ui.HTML(f'<i class="fas fa-tag"></i>'),
                    f" Categoría: {category.title()}"
                ),
                ui.div(
                    ui.HTML(f"<strong>Principales términos del cluster:</strong> "),
                    ', '.join([f'{term}' for term in cluster_terms[:6]]) if cluster_terms else "No disponibles",
                    class_="category-terms"
                ),
                class_="category-info"
            )
            result_sections.append(category_section)
            
            # Mostrar resultados de todos los modelos
            if results:
                result_sections.append(
                    ui.div(
                        ui.div(
                            ui.h3([
                                ui.HTML('<i class="fas fa-chart-line"></i>'),
                                "Resultados del análisis"
                            ], class_="panel-title"),
                            class_="panel-header"
                        ),
                        ui.div(
                            ui.div(
                                # Generar tarjeta para cada modelo
                                [ui.div(
                                    ui.div(model_name, class_="model-name"),
                                    ui.div(
                                        ui.span(result['prediction'],
                                                class_=f"prediction-badge prediction-{'fake' if result['is_fake'] else 'real'}"),
                                    ),
                                    # Barra visual de confianza
                                    ui.div(
                                        ui.HTML(f"Confianza: {create_confidence_bar(result['confidence'])}"),
                                        class_="confidence-bar"
                                    ),
                                    ui.div(f"Categoría: {result['category']}", class_="confidence-text"),

                                    # Mostrar características usadas por el modelo
                                    ui.div(
                                        ui.h6("Características usadas:", style="margin-top: 1rem; font-weight: bold;"),
                                        ui.tags.ul([
                                            ui.tags.li(f"{k}: {v}")
                                            for k, v in result.get('features', {}).items()
                                            if isinstance(v, (str, int, float)) and len(str(v)) < 80
                                        ]), 
                                        style="font-size: 0.85rem; color: var(--gray-600);"
                                    ),

                                    class_="result-card"
                                ) for model_name, result in results.items()],

                                class_="results-grid"
                            ),
                            class_="panel-body"
                        ),
                        class_="content-panel fade-in"
                    )
                )
            
            # Mostrar errores si ocurrieron
            if errors:
                result_sections.append(
                    ui.div(
                        ui.div([
                            ui.HTML('<i class="fas fa-exclamation-circle"></i>'),
                            "Errores durante el análisis"
                        ], class_="error-title"),
                        ui.div(errors, class_="error-details"),
                        class_="alert alert-error"
                    )
                )
        else:
            # =====================================================================
            # ANÁLISIS INDIVIDUAL (SOLO TÍTULO O SOLO CUERPO)
            # =====================================================================
            if title.strip():
                # Análisis solo del título
                results, errors, category, cluster_terms = make_real_prediction(
                    title, "", 'title', models, tokenizer, device,
                    kmeans_model, kmeans_vectorizer
                )
                
                # Mostrar categoría y resultados específicos para título
                category_section = ui.div(
                    ui.div(
                        ui.HTML(f'<i class="fas fa-tag"></i>'),
                        f"Categoría: {category.title()}"
                    ),
                    ui.div(
                        ui.HTML(f"<strong>Principales términos del cluster:</strong> "),
                        ', '.join([f'"{term}"' for term in cluster_terms[:6]]) if cluster_terms else "No disponibles",
                        class_="category-terms"
                    ),
                    class_="category-info"
                )
                result_sections.append(category_section)
                
                if results:
                    result_sections.append(
                        ui.div(
                            ui.div(
                                ui.h3([
                                    ui.HTML('<i class="fas fa-newspaper"></i>'),
                                    "Análisis del título"
                                ], class_="panel-title"),
                                class_="panel-header"
                            ),
                            ui.div(
                                ui.div(
                                    [ui.div(
                                        ui.div(model_name, class_="model-name"),
                                        ui.div(
                                            ui.span(result['prediction'], 
                                                   class_=f"prediction-badge prediction-{'fake' if result['is_fake'] else 'real'}"),
                                        ),
                                        ui.div(
                                            ui.HTML(f"Confianza: {create_confidence_bar(result['confidence'])}"),
                                            class_="confidence-bar"
                                        ),
                                        class_="result-card"
                                    ) for model_name, result in results.items()],
                                    class_="results-grid"
                                ),
                                class_="panel-body"
                            ),
                            class_="content-panel fade-in"
                        )
                    )
                
                if errors:
                    result_sections.append(
                        ui.div(
                            ui.div([
                                ui.HTML('<i class="fas fa-exclamation-circle"></i>'),
                                "Errores durante el análisis del título"
                            ], class_="error-title"),
                            ui.div(errors, class_="error-details"),
                            class_="alert alert-error"
                        )
                    )
            
            if body.strip():
                # =====================================================================
                # ANÁLISIS SOLO DEL CUERPO
                # =====================================================================
                results, errors, category, cluster_terms = make_real_prediction(
                    "", body, 'body', models, tokenizer, device,
                    kmeans_model, kmeans_vectorizer
                )
                
                # Mostrar categoría detectada con términos
                category_section = ui.div(
                    ui.div(
                        ui.HTML(f'<i class="fas fa-tag"></i>'),
                        f"Categoría: {category.title()}"
                    ),
                    ui.div(
                        ui.HTML(f"<strong>Principales términos del cluster:</strong> "),
                        ', '.join([f'"{term}"' for term in cluster_terms[:6]]) if cluster_terms else "No disponibles",
                        class_="category-terms"
                    ),
                    class_="category-info"
                )
                result_sections.append(category_section)
                
                if results:
                    result_sections.append(
                        ui.div(
                            ui.div(
                                ui.h3([
                                    ui.HTML('<i class="fas fa-file-text"></i>'),
                                    "Análisis del contenido"
                                ], class_="panel-title"),
                                class_="panel-header"
                            ),
                            ui.div(
                                ui.div(
                                    [ui.div(
                                        ui.div(model_name, class_="model-name"),
                                        ui.div(
                                            ui.span(result['prediction'], 
                                                   class_=f"prediction-badge prediction-{'fake' if result['is_fake'] else 'real'}"),
                                        ),
                                        ui.div(
                                            ui.HTML(f"Confianza: {create_confidence_bar(result['confidence'])}"),
                                            class_="confidence-bar"
                                        ),
                                        class_="result-card"
                                    ) for model_name, result in results.items()],
                                    class_="results-grid"
                                ),
                                class_="panel-body"
                            ),
                            class_="content-panel fade-in"
                        )
                    )
                
                if errors:
                    result_sections.append(
                        ui.div(
                            ui.div([
                                ui.HTML('<i class="fas fa-exclamation-circle"></i>'),
                                "Errores durante el análisis del contenido"
                            ], class_="error-title"),
                            ui.div(errors, class_="error-details"),
                            class_="alert alert-error"
                        )
                    )
        
        # Mostrar errores de carga de modelos si existen
        if model_loading_errors:
            result_sections.append(
                ui.div(
                    ui.div([
                        ui.HTML('<i class="fas fa-cog"></i>'),
                        "Errores de carga de modelos"
                    ], class_="error-title"),
                    ui.div("\n".join(model_loading_errors[:10]), class_="error-details"),
                    class_="alert alert-error"
                )
            )
        
        # Información educativa sobre el funcionamiento
        info_section = ui.div(
            ui.div(
                ui.HTML('''
                    <i class="fas fa-info-circle"></i>
                    El sistema primero analiza el texto de la noticia y predice su categoría temática mediante un modelo de clustering K-means. Esta categoría se utiliza como una característica adicional para mejorar la detección de desinformación mediante modelos supervisados como Naive Bayes, SVM y Random Forest.
                    '''),

                class_="alert alert-info"
            )
        )
        result_sections.append(info_section)
        
        return ui.div(*result_sections)
    
    # ==========================================================================
    # SECCIÓN DE PREPROCESAMIENTO PARA NOTICIAS
    # ==========================================================================
    @output
    @render.ui
    @reactive.event(input.btn_analyze_news)
    def news_preprocessing():
        """
        Muestra el análisis detallado paso a paso del preprocesamiento de texto.
        Solo se ejecuta para noticias, no para redes sociales.
        """
        title = input.news_title()
        body = input.news_body()
        
        # No mostrar preprocesamiento si no hay texto
        if not title.strip() and not body.strip():
            return None
        
        # Determinar qué texto procesar según lo disponible
        text_to_process = ""
        if title.strip() and body.strip():
            text_to_process = f"{title} {body}"
            process_title = "Procesamiento del texto completo"
        elif title.strip():
            text_to_process = title
            process_title = "Procesamiento del título"
        else:
            text_to_process = body
            process_title = "Procesamiento del contenido"
        
        # Obtener pasos detallados del preprocesamiento
        steps = get_preprocessing_steps(text_to_process)
        
        return ui.div(
            ui.div(
                ui.div(
                    ui.h3([
                        ui.HTML('<i class="fas fa-cogs"></i>'),
                        process_title
                    ], class_="preprocessing-title"),
                    class_="preprocessing-header"
                ),
                ui.div(
                    ui.div(
                        # Crear una tarjeta para cada paso del preprocesamiento
                        [ui.div(step, class_="preprocessing-step") for step in steps],
                        class_="preprocessing-steps"
                    ),
                    class_="preprocessing-content"
                ),
                class_="preprocessing-panel fade-in"
            )
        )
    
    # ==========================================================================
    # MANEJADOR DE ANÁLISIS DE REDES SOCIALES 
    # ==========================================================================
    @output
    @render.ui
    @reactive.event(input.btn_analyze_social)
    def social_results():
        """
        Ejecuta análisis multimodal para redes sociales.
        Requiere tanto texto como imagen para funcionar.
        """
        text = input.social_text()
        has_image = input.social_image() is not None
        
        # Validar contenido textual
        if not text.strip():
            return ui.div(
                ui.div(
                    ui.HTML('<i class="fas fa-exclamation-triangle"></i> Por favor, introduce el contenido de la publicación para analizar.'),
                    class_="alert alert-warning"
                )
            )
        
        # Validar imagen obligatoria
        if not has_image:
            return ui.div(
                ui.div(
                    ui.HTML('<i class="fas fa-image"></i> Por favor, sube una imagen para realizar el análisis multimodal.'),
                    class_="alert alert-warning"
                )
            )
        
        # Procesar imagen subida
        image_bytes = None
        try:
            file_info = input.social_image()[0]
            # Leer archivo de imagen desde el sistema temporal de Shiny
            with open(file_info["datapath"], "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            return ui.div(
                ui.div(
                    ui.HTML(f'<i class="fas fa-exclamation-circle"></i> Error procesando la imagen: {str(e)}'),
                    class_="alert alert-error"
                )
            )
        
        # Ejecutar predicción multimodal (sin K-means para redes sociales)
        results, errors, _, _ = make_social_media_prediction(
            text, image_bytes, models, device
        )
        
        result_sections = []
        
        # Mostrar resultados del análisis multimodal
        if results:
            result_sections.append(
                ui.div(
                    ui.div(
                        ui.h3([
                            ui.HTML('<i class="fas fa-layer-group"></i>'),
                            "Análisis Multimodal (Texto + Imagen)"
                        ], class_="panel-title"),
                        class_="panel-header"
                    ),
                    ui.div(
                        ui.div(
                            [ui.div(
                                ui.div(model_name, class_="model-name"),
                                ui.div(
                                    ui.span(result['prediction'],
                                            # Determinar color del badge según el tipo de predicción
                                            class_=f"prediction-badge prediction-{'real' if result['prediction'] == 'TRUE' else 'fake' if result['is_fake'] else 'other'}"),
                                ),
                                ui.div(
                                    ui.HTML(f"Confianza: {create_confidence_bar(result['confidence'])}"),
                                    class_="confidence-bar"
                                ),
                                class_="result-card"
                            ) for model_name, result in results.items()],
                            class_="results-grid"
                        ),
                        class_="panel-body"
                    ),
                    class_="content-panel fade-in"
                )
            )
        
        # Mostrar errores técnicos si ocurrieron
        if errors:
            result_sections.append(
                ui.div(
                    ui.div([
                        ui.HTML('<i class="fas fa-exclamation-circle"></i>'),
                        "Errores durante el análisis"
                    ], class_="error-title"),
                    ui.div(errors, class_="error-details"),
                    class_="alert alert-error"
                )
            )
        
        # Información educativa sobre las clases del modelo multimodal
        info_section = ui.div(
            ui.div(
                ui.HTML('''
                    <i class="fas fa-info-circle"></i>
                    <strong>Categorías del modelo multimodal</strong><br/>
                    El modelo clasifica las publicaciones en seis tipos:<br/>
                    • <strong>Verdadero</strong>: contenido auténtico y verificado.<br/>
                    • <strong>Sátira/Parodia</strong>: publicaciones con tono humorístico o irónico.<br/>
                    • <strong>Engañoso</strong>: información verdadera presentada de forma confusa o distorsionada.<br/>
                    • <strong>Suplantación</strong>: uso de fuentes o identidades falsas.<br/>
                    • <strong>Conexión falsa</strong>: el título no guarda relación con el contenido.<br/>
                    • <strong>Manipulado</strong>: contenido alterado, ya sea imagen o texto.<br/>
                    <br/>
                    El modelo analiza tanto el contenido textual como visual para emitir la predicción.
                    '''),

                class_="alert alert-info"
            )
        )
        result_sections.append(info_section)
        
        return ui.div(*result_sections)

app = App(app_ui, server)



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)

