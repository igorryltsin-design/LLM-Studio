@echo off
setlocal EnableExtensions
rem Простой помощник для Windows CMD: ставит npm-зависимости, запускает frontend и backend.

set MODE=preview
if /I "%~1"=="dev" set MODE=dev

set "BASE_HOST=127.0.0.1"
set "BASE_PORT=8001"
set "BASE_STARTED=0"
set "BASE_WINDOW=llm-base-server-%RANDOM%%RANDOM%"

cd /d "%~dp0"
set "PROJECT_DIR=%CD%"

where npm >nul 2>nul || (
    echo [Error] npm не найден. Установите Node.js и убедитесь, что npm доступен в PATH.
    endlocal & exit /b 1
)

where python >nul 2>nul || (
    echo [Error] Python не найден. Установите Python и убедитесь, что он доступен в PATH.
    endlocal & exit /b 1
)

if exist "%PROJECT_DIR%\python_libs" (
    set "PYTHONPATH=%PROJECT_DIR%\python_libs"
    set "PYTHONNOUSERSITE=1"
    set "USE_VENDOR=1"
)

echo [Info] Каталог проекта: %PROJECT_DIR%
echo [Info] Устанавливаю зависимости npm (при необходимости)...
call npm install
if errorlevel 1 goto fail

if /I "%MODE%" NEQ "dev" (
    echo [Info] Собираю production-бандл фронтенда...
    call npm run build
    if errorlevel 1 goto fail
) else (
    echo [Info] Запуск в dev-режиме — сборка фронтенда пропущена.
)

echo [Info] Запускаю сервер базовой модели на http://%BASE_HOST%:%BASE_PORT%
if defined USE_VENDOR (
    start "%BASE_WINDOW%" cmd /c "cd /d \"%PROJECT_DIR%\" && set \"PYTHONPATH=%PYTHONPATH%\" && set \"PYTHONNOUSERSITE=%PYTHONNOUSERSITE%\" && python -m uvicorn base_model_server:app --host %BASE_HOST% --port %BASE_PORT%"
) else (
    start "%BASE_WINDOW%" cmd /c "cd /d \"%PROJECT_DIR%\" && python -m uvicorn base_model_server:app --host %BASE_HOST% --port %BASE_PORT%"
)
set "BASE_STARTED=1"
timeout /t 2 /nobreak >nul

if /I "%MODE%"=="dev" (
    echo [Info] Запускаю фронтенд Vite в dev-режиме...
    call npm run dev
) else (
    echo [Info] Запускаю фронтенд Vite в режиме предпросмотра...
    call npm run preview -- --host
)
set EXITCODE=%ERRORLEVEL%
goto cleanup

:fail
set EXITCODE=1

:cleanup
if "%BASE_STARTED%"=="1" (
    echo [Info] Останавливаю сервер базовой модели...
    taskkill /FI "WINDOWTITLE eq %BASE_WINDOW%" >nul 2>nul
)
endlocal & exit /b %EXITCODE%

