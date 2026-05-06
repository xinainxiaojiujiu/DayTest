@echo off
chcp 65001 >nul 2>&1
setlocal

echo ============================================
echo   Git Upload Script v2
echo ============================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "GIT_PATH=D:\Git\bin\git.exe"

echo Step 1: Initializing Git repository...
"%GIT_PATH%" init

echo Step 2: Adding all files...
"%GIT_PATH%" add .

echo Step 3: Creating commit...
"%GIT_PATH%" commit -m "Initial commit: Automotive Coating Defect Analysis System"

echo Step 4: Setting branch name to main...
"%GIT_PATH%" branch -M main

echo Step 5: Adding remote repository...
"%GIT_PATH%" remote add origin https://github.com/xinainxiaojiujiu/DayTest.git

echo.
echo Step 6: Pushing to GitHub...
echo Please login when prompted...
"%GIT_PATH%" push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo   SUCCESS! Upload completed!
    echo   https://github.com/xinainxiaojiujiu/DayTest
    echo ============================================
) else (
    echo.
    echo ============================================
    echo   FAILED! Trying with master branch...
    echo ============================================
    "%GIT_PATH%" branch -M master
    "%GIT_PATH%" push -u origin master
)

echo.
pause
