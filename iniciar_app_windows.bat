@echo off
setlocal enabledelayedexpansion
title Configuración Automática de Python 3.11

:: 1. Configuración inicial
set PYTHON_VERSION=3.11
set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set INSTALLER=%TEMP%\python_installer.exe

:: 2. Verificar Python 3.11
echo Verificando si Python %PYTHON_VERSION% está instalado...

where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set CURRENT_VER=%%i
    echo Versión detectada: !CURRENT_VER!
    
    echo !CURRENT_VER! | findstr /C:"%PYTHON_VERSION%" >nul
    if !errorlevel! neq 0 (
        echo Versión de Python incorrecta: !CURRENT_VER!
        goto INSTALL_PYTHON
    ) else (
        echo Python %PYTHON_VERSION% ya está instalado.
        goto CONTINUE
    )
) else (
    echo Python no está instalado.
    goto INSTALL_PYTHON
)

:INSTALL_PYTHON
echo Descargando Python %PYTHON_VERSION% desde:
echo %PYTHON_URL%
curl -o "%INSTALLER%" "%PYTHON_URL%"
if exist "%INSTALLER%" (
    echo Instalando Python...
    start /wait "" "%INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    echo Instalación de Python completada.
) else (
    echo Error: No se pudo descargar el instalador de Python.
    exit /b 1
)

:: Verifica si Python se instaló correctamente
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: La instalación de Python falló.
    exit /b 1
)

:CONTINUE
:: 3. Crear entorno virtual si no existe
if not exist "venv\" (
    echo Creando entorno virtual...
    python -m venv venv
)

call venv\Scripts\activate.bat

:: 4. Instalar dependencias
if exist "requirements.txt" (
    echo Instalando dependencias desde requirements.txt...
    pip install --upgrade pip
    python --version
    pip install -r requirements.txt
)

:: 5. Ejecutar aplicación
echo Iniciando aplicación...
python app/app.py

pause
exit /b 0

:END
endlocal
