@echo off
rem ============================================================
rem Hehuang Agent dual-service launcher example (safe for GitHub)
rem   8001 = FastAPI Agent service
rem   8501 = Streamlit UI
rem
rem Usage:
rem   1) Copy this file to 启动Agent双服务.bat
rem   2) Set LLM_API_KEY in your system environment or in the copied file
rem   3) Keep the copied file private; it is ignored by .gitignore
rem ============================================================
cd /d %~dp0
set PYTHONIOENCODING=utf-8

if "%LLM_API_KEY%"=="" (
  echo [ERROR] LLM_API_KEY is not set.
  echo Please set it first, for example:
  echo   PowerShell: $env:LLM_API_KEY="sk-..."
  echo   CMD:        set LLM_API_KEY=sk-...
  pause
  exit /b 1
)

if "%LLM_BASE_URL%"=="" set LLM_BASE_URL=https://api.moonshot.cn/v1
if "%LLM_MODEL%"=="" set LLM_MODEL=kimi-k2.6
if "%LLM_TEMPERATURE%"=="" set LLM_TEMPERATURE=1
if "%EMBEDDING_PROVIDER%"=="" set EMBEDDING_PROVIDER=local
if "%LOCAL_EMBEDDING_MODEL%"=="" set LOCAL_EMBEDDING_MODEL=./models/bge-small-zh-v1.5
if "%PUPPET_API_BASE%"=="" set PUPPET_API_BASE=http://localhost:8000

echo [1/2] Starting API service (8001)...
start "Agent-API-8001" /min cmd /c "python -m uvicorn agent_project.api.server:app --host 127.0.0.1 --port 8001"
timeout /t 6 /nobreak >nul

echo [2/2] Starting UI (8501)...
start "Agent-UI-8501" cmd /c "python -m streamlit run app.py --server.headless true --server.port 8501"

echo.
echo Done. Opening the UI in your browser...
timeout /t 8 /nobreak >nul
start http://localhost:8501
pause
