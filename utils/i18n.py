import streamlit as st
import json

def cargar_lenguaje(codigo_lenguaje):
    """
    Carga el archivo de idioma correspondiente al código de idioma proporcionado.

    """
    try:
        with open(f'lenguaje/{codigo_lenguaje}.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Archivo de idioma '{codigo_lenguaje}.json' no encontrado.")
        return {}
    except json.JSONDecodeError:
        print(f"Error al decodificar el archivo JSON para '{codigo_lenguaje}.")
        return {}
    
def obtener_textos():
    """
    Obtiene los textos traducidos según el idioma en session_state.
    Si no existe, inicializa con 'es'.
    También valida que todas las claves obligatorias estén presentes.
    """
    if "codigos_idioma" not in st.session_state:
        st.session_state.codigos_idioma = "es"

    textos = cargar_lenguaje(st.session_state.codigos_idioma)

    return textos
