import json

def cargar_lenguaje(codigo_lenguaje):
    """
    Carga el archivo de idioma correspondiente al código de idioma proporcionado.

    """
    try:
        with open(f'locales/{codigo_lenguaje}.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Archivo de idioma '{codigo_lenguaje}.json' no encontrado.")
        return {}
    except json.JSONDecodeError:
        print(f"Error al decodificar el archivo JSON para '{codigo_lenguaje}.")
        return {}