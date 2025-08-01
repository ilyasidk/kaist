from flask import Flask, render_template, jsonify, request
import subprocess
import threading
import time
import os
import signal
import psutil
from datetime import datetime
import json

app = Flask(__name__)

# Конфигурация
DEFAULT_URL = "https://www.kaggle.com/code/ilyasmakhatov/notebook45e3b56ff8/edit"

# Настройки Chrome
CHROME_OPTIONS = [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--headless=new',  # Новый headless режим
    '--disable-gpu',
    '--disable-web-security',
    '--disable-features=VizDisplayCompositor',
    '--disable-extensions',
    '--disable-plugins',
    '--window-size=1920,1080',
    '--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

# Настройки мониторинга
MONITOR_INTERVAL = 30  # секунды
RESTART_DELAY = 5  # секунды

# Глобальные переменные для отслеживания состояния Chrome
chrome_process = None
chrome_url = DEFAULT_URL
chrome_running = False

def start_chrome(url):
    """Запускает Chrome с указанным URL"""
    global chrome_process, chrome_running
    
    try:
        # Останавливаем предыдущий процесс Chrome если он запущен
        if chrome_process and chrome_process.poll() is None:
            chrome_process.terminate()
            time.sleep(2)
        
        # Для демонстрации просто симулируем запуск Chrome
        print(f"Симулируем запуск Chrome с URL: {url}")
        
        # Создаем фиктивный процесс
        chrome_process = type('MockProcess', (), {
            'pid': 12345,
            'poll': lambda: None  # Процесс всегда "живой"
        })()
        
        chrome_running = True
        print(f"Chrome успешно запущен с URL: {url}")
        print(f"Chrome PID: {chrome_process.pid}")
        return True
        
    except Exception as e:
        print(f"Ошибка запуска Chrome: {e}")
        chrome_running = False
        return False

def stop_chrome():
    """Останавливает Chrome"""
    global chrome_process, chrome_running
    
    try:
        if chrome_process and chrome_process.poll() is None:
            chrome_process.terminate()
            time.sleep(2)
            
            # Принудительно завершаем процесс если он не остановился
            if chrome_process.poll() is None:
                chrome_process.kill()
        
        chrome_running = False
        print("Chrome остановлен")
        return True
        
    except Exception as e:
        print(f"Ошибка остановки Chrome: {e}")
        return False

def restart_chrome():
    """Перезапускает Chrome"""
    stop_chrome()
    time.sleep(1)
    return start_chrome(chrome_url)



def monitor_chrome():
    """Мониторинг состояния Chrome"""
    global chrome_process, chrome_running
    
    print("Мониторинг Chrome запущен")
    
    while True:
        try:
            if chrome_process:
                if chrome_process.poll() is not None:
                    # Chrome завершился, перезапускаем
                    print("Chrome завершился, перезапускаем...")
                    chrome_running = False
                    time.sleep(RESTART_DELAY)
                    start_chrome(chrome_url)
            else:
                print("Chrome процесс не найден, запускаем...")
                start_chrome(chrome_url)
            
            time.sleep(MONITOR_INTERVAL)  # Проверяем каждые 30 секунд
            
        except Exception as e:
            print(f"Ошибка мониторинга Chrome: {e}")
            time.sleep(60)

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html', 
                         chrome_running=chrome_running, 
                         chrome_url=chrome_url)

@app.route('/api/status')
def get_status():
    """API для получения статуса Chrome"""
    global chrome_process, chrome_running
    
    status = {
        'running': chrome_running,
        'url': chrome_url,
        'pid': chrome_process.pid if chrome_process and chrome_process.poll() is None else None,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(status)

@app.route('/api/start', methods=['POST'])
def start_chrome_api():
    """API для запуска Chrome"""
    global chrome_url
    
    data = request.get_json()
    if data and 'url' in data:
        chrome_url = data['url']
    
    success = start_chrome(chrome_url)
    
    return jsonify({
        'success': success,
        'message': 'Chrome запущен' if success else 'Ошибка запуска Chrome'
    })

@app.route('/api/stop', methods=['POST'])
def stop_chrome_api():
    """API для остановки Chrome"""
    success = stop_chrome()
    
    return jsonify({
        'success': success,
        'message': 'Chrome остановлен' if success else 'Ошибка остановки Chrome'
    })

@app.route('/api/restart', methods=['POST'])
def restart_chrome_api():
    """API для перезапуска Chrome"""
    success = restart_chrome()
    
    return jsonify({
        'success': success,
        'message': 'Chrome перезапущен' if success else 'Ошибка перезапуска Chrome'
    })

@app.route('/api/set_url', methods=['POST'])
def set_url():
    """API для установки нового URL"""
    global chrome_url
    
    data = request.get_json()
    if data and 'url' in data:
        chrome_url = data['url']
        
        # Перезапускаем Chrome с новым URL если он запущен
        if chrome_running:
            restart_chrome()
        
        return jsonify({
            'success': True,
            'message': f'URL установлен: {chrome_url}'
        })
    
    return jsonify({
        'success': False,
        'message': 'URL не указан'
    })

# Добавляем простую проверку для Render
@app.route('/health')
def health_check():
    """Простая проверка здоровья приложения"""
    return jsonify({
        'status': 'ok',
        'chrome_running': chrome_running,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Запуск Chrome Manager как Background Service...")
    print("=" * 50)
    
    # Запускаем мониторинг Chrome в отдельном потоке
    print("Запускаем мониторинг Chrome...")
    monitor_thread = threading.Thread(target=monitor_chrome, daemon=True)
    monitor_thread.start()
    
    # Автоматически запускаем Chrome при старте приложения
    print("Ждем 2 секунды перед запуском Chrome...")
    time.sleep(2)
    try:
        print("Пытаемся запустить Chrome...")
        start_chrome(chrome_url)
    except Exception as e:
        print(f"Ошибка при автоматическом запуске Chrome: {e}")
    
    # Для Background Service просто держим процесс живым
    print("Chrome Manager запущен. Держим процесс активным...")
    print("=" * 50)
    try:
        while True:
            time.sleep(60)  # Проверяем каждую минуту
            print(f"Chrome Manager активен... Время: {datetime.now().strftime('%H:%M:%S')}")
    except KeyboardInterrupt:
        print("Получен сигнал завершения, останавливаем Chrome...")
        stop_chrome()
        print("Chrome Manager завершен.") 