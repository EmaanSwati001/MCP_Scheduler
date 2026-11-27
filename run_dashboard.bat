@echo off
cd /d "%~dp0"
echo Starting MCG-Sched Control Plane at http://127.0.0.1:8501 ...
python -m dashboard.server
pause
