@echo off
setlocal enabledelayedexpansion

:: ============================================
:: Automotive Coating Defect Analysis System
:: One-click startup script
:: ============================================

title Automotive Coating Defect Analysis System

echo.
echo ============================================
echo    Automotive Coating Defect Analysis v1.0.0
echo ============================================
echo.
echo    [TIP] First run will auto-install dependencies
echo    Access http://localhost:8000/docs for API docs
echo ============================================
echo.

set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
cd /d "%~dp0"

:: Step 1: Check Python
echo [1/4] Checking Python environment...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.9+ from:
    echo https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo    [OK] Python %PYVER%

:: Step 2: Check .env config
echo [2/4] Checking configuration...

if not exist ".env" (
    echo.
    echo    [INFO] .env not found, creating from template...
    if exist ".env.example" (
        copy .env.example .env >nul 2>&1
        echo    [OK] .env created from .env.example
        echo.
        echo    [IMPORTANT] Please edit .env file and set at least one LLM API key:
        echo      - Alibaba DashScope: set DASHSCOPE_API_KEY
        echo      - Dify Platform: set DIFY_API_KEY and change LLM_PLATFORM to dify
        echo.
        echo    Then run this script again.
        echo.
        pause
        exit /b 0
    ) else (
        echo    [WARN] .env.example not found, using defaults
    )
) else (
    echo    [OK] .env configuration file found
)

:: Step 3: Check dependencies
echo [3/4] Checking Python packages...

set NEED_INSTALL=0

python -c "import fastapi" 2>nul || set NEED_INSTALL=1
python -c "import uvicorn" 2>nul || set NEED_INSTALL=1
python -c "import pydantic_settings" 2>nul || set NEED_INSTALL=1
python -c "import sqlalchemy" 2>nul || set NEED_INSTALL=1
python -c "import httpx" 2>nul || set NEED_INSTALL=1
python -c "import tenacity" 2>nul || set NEED_INSTALL=1
python -c "import dotenv" 2>nul || set NEED_INSTALL=1
python -c "import numpy" 2>nul || set NEED_INSTALL=1

if %NEED_INSTALL% equ 1 (
    echo.
    echo    [INFO] Missing packages detected, installing...
    echo    [INFO] This may take a few minutes, please wait...
    echo.
    pip install fastapi "uvicorn[standard]" "sqlalchemy[asyncio]" aiomysql httpx tenacity python-dotenv numpy pydantic-settings -q
    if %errorlevel% neq 0 (
        echo.
        echo    [WARN] Install failed, trying mirror source...
        pip install fastapi "uvicorn[standard]" "sqlalchemy[asyncio]" aiomysql httpx tenacity python-dotenv numpy pydantic-settings -q -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
    echo    [OK] Dependencies installed
) else (
    echo    [OK] Core packages ready
)

:: Step 4: Start server
echo [4/4] Starting server...
echo.
echo ============================================
echo    Starting system...
echo ============================================
echo.
echo    API Server:     http://localhost:8000
echo    API Docs:       http://localhost:8000/docs
echo    Homepage:       http://localhost:8000/
echo.
echo    Press Ctrl+C to stop
echo ============================================
echo.

python -m src.main

echo.
echo ============================================
echo    System stopped
echo ============================================
pause
