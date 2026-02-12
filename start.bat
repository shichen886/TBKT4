@echo off
cd /d "%~dp0"
C:/Users/32880/miniconda3/envs/emnist-gpu/python.exe -m streamlit run app.py --browser.gatherUsageStats false
pause
