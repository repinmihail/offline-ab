@echo off
echo Build Docker...
"C:\Program Files\Docker\Docker\resources\bin\docker.exe" build -t offline-ab .

if %errorlevel% == 0 (
    echo Run container...
    "C:\Program Files\Docker\Docker\resources\bin\docker.exe" run --rm offline-ab
) else (
    echo Error, check logs above
)