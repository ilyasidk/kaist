#!/bin/bash

# Скрипт для установки Google Chrome на сервере Render
# Этот скрипт будет выполнен во время сборки

echo "Установка Google Chrome..."

# Обновляем пакеты
apt-get update

# Устанавливаем зависимости
apt-get install -y wget gnupg2

# Добавляем репозиторий Google Chrome
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list

# Обновляем пакеты и устанавливаем Chrome
apt-get update
apt-get install -y google-chrome-stable

# Проверяем установку
google-chrome --version

echo "Chrome установлен успешно!"

# Создаем директорию для данных Chrome
mkdir -p /tmp/chrome-data
chmod 755 /tmp/chrome-data

echo "Настройка завершена!" 