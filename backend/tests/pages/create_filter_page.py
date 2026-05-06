# pages/create_filter_page.py
from selenium.webdriver.common.by import By
from .base_page import BasePage

class CreateFilterPage(BasePage):

    URL = "http://frontend:3000/"

    BUTTON_TO_CREATE_FILTER = (By.XPATH, "//button[text()='Create New Style']")
    HEADING_OF_CREATE_FILTER = (By.XPATH, "//h2[text()='Create New Makeup Style']")
    PANEL_REFERENCE_IMAGE = (By.XPATH, "//p[text()='Drag & drop the makeup reference (style) image']")
    PANEL_MODEL_IMAGE = (By.XPATH, "//p[text()='Optional: Drag & drop a model image for preview']")
    BUTTON_CREATE_STYLE = (By.XPATH, "//button[text()='Create New Style']")
    BUTTON_LIVE_CAMERA = (By.XPATH, "//button[text()='Live camera']")

    def open_create_filter_page(self):
        self.driver.get(self.URL)

    def upload_reference_image(self, path):
        self.find(self.PANEL_REFERENCE_IMAGE).send_keys(path)

    def upload_model_image(self, path):
        self.find(self.PANEL_MODEL_IMAGE).send_keys(path)

    def click_button_to_create_filter(self):
        self.click(self.BUTTON_TO_CREATE_FILTER)

    def click_panel_reference_image(self):
        self.click(self.PANEL_REFERENCE_IMAGE)

    def click_button_create_style(self):
        self.click(self.BUTTON_CREATE_STYLE)

    def find_heading(self):
        return self.find(self.HEADING_OF_CREATE_FILTER).text