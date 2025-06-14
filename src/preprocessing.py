import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import re
import spacy

# Solo necesitas ejecutar esto una vez
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from typing import Any

def recortar_texto(texto: Any, max_palabras: int = 300) -> str:
    """
    Recorta el texto a un máximo de `max_palabras` palabras.

    Parámetros:
    texto: El contenido que se desea truncar; se convertirá a str si no lo es.
    max_palabras: Número máximo de palabras a conservar (por defecto 300).

    Devuelve:
    El texto recortado a las primeras `max_palabras` palabras.
    """
    # Convertimos a str cualquier entrada que no lo sea
    texto_str = str(texto)
    # split con maxsplit para no partir todo el string más allá de lo necesario
    palabras = texto_str.split(maxsplit=max_palabras)
    # Unimos sólo las primeras max_palabras
    return " ".join(palabras[:max_palabras])
""" """ """ """ """  """ """ """ """ """


# Inicializamos objetos necesarios
""" """  """ """
stop_words = set(stopwords.words('english'))
# Stopwords en inglés pero mantener algunas importantes
stop_words = set(stopwords.words('english'))

# Mantener palabras que pueden ser importantes para fake news
keep_words = {
    'not', 'no', 'never', 'always', 'all', 'every', 
    'very', 'much', 'more', 'most', 'never', 'nothing',
    'everything', 'everyone', 'nobody', 'none'
}

stop_words = stop_words - keep_words

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if not isinstance(text, str):
        return ""

    # Minúsculas
    text = text.lower()

    # Eliminar URLs y correos
    text = re.sub(r"http\S+|www\S+|https\S+", " urltoken ", text)
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)

    # Sustituir exclamaciones e interrogaciones múltiples
    text = re.sub(r'!{2,}', ' strongexclamationtoken ', text)  # !! o más
    text = re.sub(r'\?{2,}', ' strongquestiontoken ', text)    # ?? o más
    text = re.sub(r'[!?]{3,}', ' emphasistoken ', text)        # Mezclas como !?!?
    
    # Sustituir exclamaciones e interrogaciones simples
    text = re.sub(r'!', ' exclamationtoken ', text)
    text = re.sub(r'\?', ' questiontoken ', text)

    # Eliminar caracteres no alfanuméricos (emojis, símbolos, emoticonos)
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenización
    tokens = word_tokenize(text)

    # Eliminar stopwords y lematizar
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return ' '.join(tokens)


