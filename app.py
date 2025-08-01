from flask import Flask, render_template, jsonify, request
import subprocess
import threading
import time
import os
import signal
import psutil
from datetime import datetime
import json
from config import *

app = Flask(__name__)

# Глобальные переменные для отслеживания состояния Chrome
chrome_process = None
chrome_url = DEFAULT_URL
chrome_running = False

def start_chrome(url):
    """Запускает Chrome с указанным URL и автоматической авторизацией"""
    global chrome_process, chrome_running
    
    try:
        # Останавливаем предыдущий процесс Chrome если он запущен
        if chrome_process and chrome_process.poll() is None:
            chrome_process.terminate()
            time.sleep(2)
        
        # Команда для запуска Chrome с авторизацией Kaggle
        chrome_cmd = ['google-chrome'] + CHROME_OPTIONS + [url]
        
        # Запускаем Chrome
        chrome_process = subprocess.Popen(chrome_cmd)
        
        chrome_running = True
        print(f"Chrome запущен с URL: {url}")
        
        # Запускаем автоматическую авторизацию в отдельном потоке
        auth_thread = threading.Thread(target=auto_login_kaggle, daemon=True)
        auth_thread.start()
        
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

def auto_login_kaggle():
    """Автоматическая авторизация на Kaggle"""
    import requests
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    
    try:
        print("Начинаем автоматическую авторизацию на Kaggle...")
        
        # Настройки Chrome для автоматизации
        chrome_options = Options()
        for option in CHROME_OPTIONS:
            chrome_options.add_argument(option)
        
        # Подключаемся к уже запущенному Chrome
        driver = webdriver.Chrome(options=chrome_options)
        
        # Переходим на страницу входа Kaggle
        driver.get("https://www.kaggle.com/account")
        
        # Ждем появления формы входа
        wait = WebDriverWait(driver, 10)
        
        # Ищем поля для ввода логина и пароля
        try:
            username_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
            password_field = driver.find_element(By.NAME, "password")
            
            # Вводим данные
            username_field.clear()
            username_field.send_keys(KAGGLE_USERNAME)
            
            password_field.clear()
            password_field.send_keys(KAGGLE_PASSWORD)
            
            # Нажимаем кнопку входа
            login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            print("Авторизация выполнена успешно!")
            
            # Ждем загрузки и переходим к нужной странице
            time.sleep(5)
            driver.get(chrome_url)
            
        except Exception as e:
            print(f"Ошибка при авторизации: {e}")
            # Если авторизация не удалась, просто переходим к URL
            driver.get(chrome_url)
        
        # Держим браузер открытым
        while chrome_running:
            time.sleep(10)
            
    except Exception as e:
        print(f"Ошибка автоматической авторизации: {e}")

def monitor_chrome():
    """Мониторинг состояния Chrome"""
    global chrome_process, chrome_running
    
    while True:
        try:
            if chrome_process:
                if chrome_process.poll() is not None:
                    # Chrome завершился, перезапускаем
                    print("Chrome завершился, перезапускаем...")
                    chrome_running = False
                    time.sleep(RESTART_DELAY)
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

if __name__ == '__main__':
    # Запускаем мониторинг Chrome в отдельном потоке
    monitor_thread = threading.Thread(target=monitor_chrome, daemon=True)
    monitor_thread.start()
    
    # Автоматически запускаем Chrome при старте приложения
    time.sleep(2)
    start_chrome(chrome_url)
    
    # Запускаем Flask приложение
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 