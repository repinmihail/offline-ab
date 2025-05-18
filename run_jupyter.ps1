# run-jupyter.ps1

$IMAGE_NAME = "offline-ab"

# Собираем образ, если его нет
$buildOutput = docker inspect --type=image $IMAGE_NAME 2>&1 | Out-String

if ($buildOutput -like "*No such image*") {
    Write-Host "🏗️ Образ не найден. Собираю Docker-образ..."
    docker build -t $IMAGE_NAME .
}

# Запускаем Jupyter
Write-Host "🚀 Запускаем Jupyter Notebook..."

$dockerRunCommand = "docker run --rm -p 8888:8888 -v ${PWD}:/app -w /app $IMAGE_NAME jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
Invoke-Expression $dockerRunCommand