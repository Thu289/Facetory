from tests.core.driver import get_driver
from tests.pages.create_filter_page import CreateFilterPage

def test_create_filter():
    driver = get_driver()

    try:
        page = CreateFilterPage(driver)

        page.open_create_filter_page()
        page.click_button_to_create_filter()
        page.find_heading() == "Create New Makeup Style"
        page.upload_reference_image("/images/00059.png")
        page.upload_model_image("/images/02277.png")
        page.click_button_create_style()
        page.click(page.BUTTON_LIVE_CAMERA)
        print("Test passed")

    finally:
        driver.quit()


if __name__ == "__main__":
    test_create_filter()