@echo off
echo Build Docker...
docker build -t offline-ab .

if %errorlevel% == 0 (
    echo Run container...
    docker run --rm offline-ab
) else (
    echo Error, check logs above
)