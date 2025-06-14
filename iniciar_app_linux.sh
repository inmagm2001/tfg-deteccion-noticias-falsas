#!/bin/bash

echo "========================================"
echo "Iniciando la aplicación Shiny para Python"
echo "========================================"
echo

# Verifica si existe el entorno virtual
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
    echo "Entorno virtual activado."
    echo
else
    echo "No se encontró entorno virtual. Creando uno nuevo..."
    python3 -m venv venv
    echo "Entorno virtual creado."
    echo "Activando entorno virtual..."
    source venv/bin/activate
    echo
fi

# Instala o actualiza las dependencias desde requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
    echo "Dependencias instaladas correctamente."
    echo
else
    echo "ADVERTENCIA: No se encontró el archivo requirements.txt"
    echo
fi

# Ejecuta la aplicación
echo "Iniciando aplicación Shiny..."
python3 app/app.py

echo
echo "========================================"
echo "Aplicación finalizada."
echo "========================================"
