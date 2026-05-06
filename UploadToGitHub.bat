@echo off
chcp 65001 >nul 2>&1
setlocal

echo ============================================
echo   Git Upload Script
echo ============================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "GIT_PATH=D:\Git\bin\git.exe"

echo [1/5] Initializing Git repository...
"%GIT_PATH%" init
if %errorlevel% neq 0 (
    echo [ERROR] Git init failed! Make sure Git is installed at D:\Git
    pause
    exit /b 1
)

echo [2/5] Adding files to staging area...
"%GIT_PATH%" add .
if %errorlevel% neq 0 (
    echo [ERROR] Git add failed!
    pause
    exit /b 1
)

echo [3/5] Creating initial commit...
"%GIT_PATH%" commit -m "Initial commit: Automotive Coating Defect Analysis System"
if %errorlevel% neq 0 (
    echo [ERROR] Git commit failed!
    pause
    exit /b 1
)

echo [4/5] Adding remote repository...
"%GIT_PATH%" remote remove origin >nul 2>&1
"%GIT_PATH%" remote add origin https://github.com/xinainxiaojiujiu/DayTest.git
if %errorlevel% neq 0 (
    echo [ERROR] Git remote add failed!
    pause
    exit /b 1
)

echo [5/5] Pushing to GitHub...
echo.
echo If prompted for login, please login to your GitHub account.
echo.

"%GIT_PATH%" push -u origin main --force

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo   [SUCCESS] Upload completed!
    echo   Repository: https://github.com/xinainxiaojiujiu/DayTest
    echo ============================================
) else (
    echo.
    echo ============================================
    echo   [WARNING] Push failed. Please check:
    echo   1. Internet connection
    echo   2. GitHub login status
    echo   3. Repository URL is correct
    echo ============================================
)

echo.
pause
