@echo off
set PORT=9999
set HOST=0.0.0.0

uvicorn main:app --host %HOST% --port %PORT% --forwarded-allow-ips '*'