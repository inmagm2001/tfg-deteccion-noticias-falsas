# Aplicación Shiny para Python

Esta aplicación web está desarrollada con Shiny para Python y proporciona una interfaz interactiva para el análisis de datos.

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Instalación y Ejecución

### Windows

#### Método automático (recomendado)

1. Clona o descarga este repositorio
2. Asegúrate de tener el archivo `requirements.txt` en la carpeta del proyecto
3. Ejecuta el archivo `iniciar_app.bat` haciendo doble clic

El script automáticamente se encarga de crear el entorno virtual, instalar las dependencias y ejecutar la aplicación.

#### Método manual

Abre la línea de comandos (CMD) o PowerShell:

```cmd
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python app.py
```

### Linux/macOS

#### Método automático

1. Dar permisos de ejecución al script:
```bash
chmod +x iniciar_app.sh
```

2. Ejecutar el script:
```bash
./iniciar_app.sh
```

#### Método manual

Abrir terminal:

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python app.py
```

## Estructura del Proyecto

```
proyecto/
├── app.py                 # Archivo principal de la aplicación
├── requirements.txt       # Lista de dependencias
├── iniciar_app.bat       # Script para Windows
├── iniciar_app.sh        # Script para Linux/macOS
├── README.md             # Documentación
└── venv/                 # Entorno virtual (generado automáticamente)
```

## Configuración de Dependencias

El archivo `requirements.txt` incluiye lo necesario:


## Acceso a la Aplicación

Una vez iniciada la aplicación, estará disponible en el navegador web en:

```
http://localhost:8000
```

La dirección exacta se mostrará en la consola al iniciar.

## Detener la Aplicación

Para finalizar la ejecución, presiona `Ctrl + C` en la ventana de comandos o terminal.

## Resolución de Problemas

### "python no se reconoce como comando"

**Windows:**
- Verificar que Python esté instalado y en el PATH del sistema
- Intentar con `py` en lugar de `python`

**Linux/macOS:**
- Usar `python3` en lugar de `python`
- Instalar Python si es necesario: `sudo apt install python3 python3-pip`

### Error al activar entorno virtual

**Windows:**
- En PowerShell, ejecutar: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Linux/macOS:**
- Verificar permisos del script: `chmod +x iniciar_app.sh`

### ModuleNotFoundError

- Confirmar que el entorno virtual esté activado
- Verificar que `requirements.txt` contenga todas las dependencias
- Ejecutar manualmente `pip install -r requirements.txt`


