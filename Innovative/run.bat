@echo off
REM ============================================================
REM  StartupAdvisorLM launcher
REM  Double-click this file (or run: run.bat) to start the app.
REM  The Groq key is read from the GROQ_API_KEY environment
REM  variable (set it once with:  setx GROQ_API_KEY your-key).
REM ============================================================

cd /d "%~dp0"

if "%GROQ_API_KEY%"=="" (
  echo [!] GROQ_API_KEY is not set - Smart Mode will be OFF.
  echo     Set it once with:  setx GROQ_API_KEY your-key
  echo     then open a NEW terminal.
  echo.
) else (
  echo [ok] Groq key detected - Smart Mode available.
  echo.
)

echo Starting StartupAdvisorLM at http://localhost:8000
echo Press CTRL+C to stop.
echo.

python -m uvicorn api:app --port 8000
