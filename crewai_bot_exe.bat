@echo off

:: Activate the conda environment
call conda activate helpshift-env
timeout /t 0 >nul  # Wait for 3 seconds

:: Start the Single Intent Agent and keep the terminal open
start cmd /k "python crew_ai_bot.py"
timeout /t 1 >nul  # Wait for 3 seconds

:: Open index.html in the default web browser
start index.html
timeout /t 3 >nul  # Wait for 3 seconds