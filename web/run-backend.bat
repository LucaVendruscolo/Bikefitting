@echo off
echo Starting BikeFit Pro Backend...
cd /d "%~dp0backend"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause

