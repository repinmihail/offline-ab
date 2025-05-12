# Базовый образ
FROM python:3.10-slim

# Установка Poetry 1.8.3
RUN pip install --no-cache-dir poetry==1.8.3

RUN pip show poetry poetry-core

# Рабочая директория
WORKDIR /app

# Сначала копируем метаданные (для лучшего кэширования)
COPY pyproject.toml README.md ./

# Копируем исходники
COPY offline_ab ./offline_ab

# Устанавливаем зависимости и сам проект
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Точка входа
CMD ["python", "-c", "import importlib.metadata; print(importlib.metadata.version('offline-ab'))"]