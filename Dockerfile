# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем только необходимые зависимости
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Создаем директорию для данных Chrome
RUN mkdir -p /tmp/chrome-data && chmod 755 /tmp/chrome-data

# Открываем порт
EXPOSE 5000

# Запускаем приложение как background service
CMD ["python", "app.py"] 