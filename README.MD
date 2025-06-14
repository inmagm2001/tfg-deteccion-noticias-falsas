# Aplicación Shiny en Python

Esta aplicación ha sido desarrollada con [Shiny para Python](https://shiny.posit.co/py/) y proporciona una interfaz interactiva para la clasificación automática de noticias falsas.

---

## Requisitos previos

### Windows

No es necesario tener Python ni pip instalados manualmente. El script `iniciar_app.bat` se encarga de comprobar si hay una instalación de Python 3.11, descargarla en caso de no estar disponible, crear el entorno virtual, instalar las dependencias y ejecutar la aplicación.

### Linux/macOS

Es necesario contar con una instalación previa de Python 3 y pip. Se recomienda verificarlo mediante los siguientes comandos:

```bash
python3 --version
pip --version
```

---

## Instalación y ejecución

### Windows

#### Método automático

1. Clonar o descargar este repositorio.
2. Asegurarse de que `requirements.txt` se encuentra en el directorio raíz del proyecto.
3. Ejecutar el archivo `iniciar_app.bat` con doble clic.

El script creará el entorno virtual, instalará las dependencias y lanzará la aplicación.

#### Método manual

Abrir CMD o PowerShell y ejecutar:

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app/app.py
```

---

### Linux/macOS

#### Método automático

1. Dar permisos de ejecución al script:

```bash
chmod +x iniciar_app.sh
```

2. Ejecutarlo desde la terminal:

```bash
./iniciar_app.sh
```

#### Método manual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app/app.py
```

---

## Estructura del proyecto

```
tfg_fakenews/
├── app/ # Código fuente de la aplicación
├── data/ # Datos usados en el proyecto
├── models/ # Modelos entrenados y guardados
├── notebooks/ # Notebooks para análisis y experimentos
├── src/ # Código fuente adicional
├── requirements.txt # Archivo con las dependencias del proyecto
├── iniciar_app_linux.sh # Script para iniciar la app en Linux/macOS
├── iniciar_app_windows.bat# Script para iniciar la app en Windows
├── README.md # Documentación del proyecto
```

---

## Acceso a la aplicación

Una vez iniciada, la aplicación estará disponible en el navegador en:

```
https://127.0.0.1:8000
```

La dirección exacta se mostrará en consola tras el arranque.

---

## Finalización de la ejecución

Para detener la aplicación, basta con presionar `Ctrl + C` en la terminal o ventana de comandos.

---

## Resolución de problemas

### "python no se reconoce como comando"

**Windows**:  
Comprobar que Python está correctamente instalado y añadido al PATH. También se puede intentar con el comando `py`.

En caso necesario, instalar Python y pip con:

```bash
sudo apt install python3 python3-pip
```

---

### Error al activar el entorno virtual

**Windows (PowerShell)**:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/macOS**:  
Verificar que el script tiene permisos de ejecución con `chmod +x`.

---

### `ModuleNotFoundError`

- Comprobar que el entorno virtual ha sido activado.
- Asegurarse de haber ejecutado `pip install -r requirements.txt`.

---

## Licencia

Este proyecto se distribuye bajo los términos de la licencia MIT.
