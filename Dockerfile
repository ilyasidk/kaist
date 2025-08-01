# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    gpg \
    curl \
    unzip \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Google Chrome и Chromium
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable chromium-browser \
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