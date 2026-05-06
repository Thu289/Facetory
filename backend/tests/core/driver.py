from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

def get_driver():
    selenium_url = os.getenv('SELENIUM_URL', 'http://selenium:4444/wd/hub')

    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Remote(
        command_executor=selenium_url,
        options=options
    )
    return driver