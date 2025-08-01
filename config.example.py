# Пример конфигурации для Chrome Manager
# Скопируйте этот файл как config.py

# URL по умолчанию
DEFAULT_URL = "https://www.kaggle.com/code/ilyasmakhatov/notebook45e3b56ff8/edit"

# Настройки Chrome
CHROME_OPTIONS = [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--remote-debugging-port=9222',
    '--user-data-dir=/tmp/chrome-data',
    '--disable-web-security',
    '--disable-features=VizDisplayCompositor',
    '--disable-extensions',
    '--disable-plugins'
]

# Настройки мониторинга
MONITOR_INTERVAL = 30  # секунды
RESTART_DELAY = 5  # секунды 