@echo off
chcp 65001 > nul
cd /d "%~dp0"

python -m streamlit run app_video.py
if errorlevel 1 (
    echo.
    echo [오류] 실행 실패. streamlit 설치를 시도합니다...
    pip install streamlit
    python -m streamlit run app_video.py
)
pause
