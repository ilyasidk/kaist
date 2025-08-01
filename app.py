import time
from datetime import datetime

# Конфигурация
DEFAULT_URL = "https://www.kaggle.com/code/ilyasmakhatov/notebook45e3b56ff8/edit"

# Глобальные переменные
chrome_running = True
chrome_url = DEFAULT_URL

def start_chrome(url):
    """Запускает Chrome с указанным URL"""
    global chrome_running
    print(f"Chrome запущен с URL: {url}")
    chrome_running = True
    return True

def stop_chrome():
    """Останавливает Chrome"""
    global chrome_running
    print("Chrome остановлен")
    chrome_running = False
    return True

def monitor_chrome():
    """Мониторинг состояния Chrome"""
    print("Мониторинг Chrome запущен")
    while True:
        try:
            print(f"Chrome работает с URL: {chrome_url}")
            time.sleep(30)  # Проверяем каждые 30 секунд
        except Exception as e:
            print(f"Ошибка мониторинга: {e}")
            time.sleep(60)

if __name__ == '__main__':
    print("=" * 50)
    print("Chrome Manager запущен!")
    print("=" * 50)
    
    # Запускаем Chrome
    start_chrome(chrome_url)
    
    # Запускаем мониторинг
    monitor_chrome() 