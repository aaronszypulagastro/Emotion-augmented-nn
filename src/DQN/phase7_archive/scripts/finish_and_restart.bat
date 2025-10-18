@echo off
echo ================================================================
echo   PHASE 7.0 - Abschluss und Neustart mit PSA
echo ================================================================
echo.

echo [1/4] Warte 3 Minuten bis Training fertig ist...
timeout /t 180 /nobreak

echo.
echo [2/4] Sichere alte Logs (Option A - ohne PSA)...
copy results\training_log.csv results\training_log_option_a_backup.csv
echo    Backup erstellt: training_log_option_a_backup.csv

echo.
echo [3/4] Loesche alte training_log.csv fuer neues Training...
del results\training_log.csv
echo    Alte Logs entfernt

echo.
echo [4/4] Starte NEUES Training MIT PSA (Option B)...
echo    Performance Stability Analyzer: AKTIV
echo    Erwartung: Verbesserte Stabilitaet
echo.
python training\train_finetuning.py

echo.
echo ================================================================
echo   FERTIG! Vergleiche Ergebnisse mit: python comprehensive_analysis.py
echo ================================================================
pause

