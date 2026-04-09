@echo off
REM 欣興 3037 價格警報：跌破 534.6 或站回 570.24 時發送 LINE
REM 可加入 Windows 工作排程，盤中每 5 分鐘執行一次
cd /d "%~dp0"
python price_alert_monitor.py 3037.TW --stop 534.6 --entry 570.24 --send-line
