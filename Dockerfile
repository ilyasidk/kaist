# Используем официальный образ Python
FROM python:3.9-slim

# Создаем рабочую директорию
WORKDIR /app

# Копируем код приложения
COPY . .

# Запускаем приложение
CMD ["python", "app.py"] 