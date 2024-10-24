from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import requests
import re

# Function to sanitize filenames
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)



# Initialize Chrome driver
driver = webdriver.Chrome()

# Define the URL to visit (search results page)
url ='https://digitalcollections.clemson.edu/search-results/?k=identifier%3Acu%2A&pagetokens=xapian-YToxOntpOjA7YTo1OntzOjM6ImNvcCI7TjtzOjI6Im9wIjtOO3M6NToidmFsdWUiO3M6MzoiY3UqIjtzOjQ6InR5cGUiO3M6MjoiUFQiO3M6NToiZmllbGQiO3M6MTA6ImlkZW50aWZpZXIiO319%7EczowOiIiOw%3D%3D%7EYToxOntpOjA7czo1OiJmYWNldCI7fQ%3D%3D%7EYToxOntpOjA7czoxOiIqIjt9%7EYTowOnt9%7E100%7Edate%7E10099'
download_dir = "downloaded_images"
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Open the URL
driver.get(url)

# Wait for the search results page to load
wait = WebDriverWait(driver, 10)

try:
    # Loop over pages and images
    for page_num in range(100):  # Set a high number to loop through pages
        # Get all result rows
        results = driver.find_elements(By.CSS_SELECTOR, '.row.item')

        for result in results:
            # Check if there is a date in the "item-results-date" div
            try:
                date_element = result.find_element(By.CSS_SELECTOR, '.item-results-date')
                date_text = date_element.text.strip()

                # Check for "View on map" link
                if result.find_elements(By.CSS_SELECTOR, "a.opnsearchmaplink") and date_text == "View on map":
                    print("Skipping image download due to 'View on map' link and no date")

                    continue  # Skip the current iteration and move to the next result

                # Only proceed if there is a date
                if date_text:
                    # Get the link to the image's page
                    link_element = result.find_element(By.CSS_SELECTOR, 'a.opnsearchitemlink')
                    image_page_url = link_element.get_attribute('href')

                    # Open the image page in a new tab
                    driver.execute_script(f"window.open('{image_page_url}');")
                    driver.switch_to.window(driver.window_handles[1])

                    # Wait for the image page to load
                    time.sleep(2)

                    # Extract the download URL from the 'Download File' link
                    download_link = driver.find_element(By.CSS_SELECTOR, '#download-links a')
                    download_url = download_link.get_attribute('href')

                    # Ensure the full download URL is created
                    if download_url.startswith("//"):
                        download_url = "https:" + download_url

                    # Extract metadata (Title and Digital Identifier)
                    title = driver.find_element(By.CSS_SELECTOR, '#metadata-header h1').text.strip()

                    # Locate the Digital Identifier using CSS selector
                    digital_identifier_element = driver.find_element(By.CSS_SELECTOR, ".metadata-item span.metadata-label + div")
                    digital_identifier = digital_identifier_element.text.strip().split()[-1].replace("Created", "")  # Get the last part (the identifier number)


                    # Combine the title and digital identifier for file naming
                    # Combine the title and digital identifier for file naming, sanitizing both parts
                    title_with_identifier = f"{sanitize_filename(title)}_{sanitize_filename(digital_identifier)}"


                    # Write the date to an individual text file for each image
                    txt_file_path = os.path.join(download_dir, f"{title_with_identifier}.txt")
                    with open(txt_file_path, 'w') as txt_file:
                        txt_file.write(f"{date_text.split()[-1]}")

                    # Download the image
                    image_path = os.path.join(download_dir, f"{title_with_identifier}.jpg")
                    
                    # Download the image using requests
                    response = requests.get(download_url)
                    with open(image_path, 'wb') as img_file:
                        img_file.write(response.content)
                    print(f"Downloaded {title_with_identifier}.jpg and saved {title_with_identifier}.txt with date: {date_text}.")

                    # Close the image tab and switch back to the main search page
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

            except Exception as e:
                print(f"Error processing result: {e}")

        # Navigate to the next page
        # Navigate to the next page
        try:
            # Locate the next page button using its image source
            next_button = driver.find_element(By.XPATH, "//img[@src='https://digitalcollections.clemson.edu/wp-content/themes/cureptheme/images/btn_arrow_right.png']/ancestor::a")
            
            # Scroll into view and click the next button
            driver.execute_script("arguments[0].scrollIntoView();", next_button)  # Scroll into view
            next_button.click()  # Click the next button
        except Exception as e:
            print(f"Error navigating to the next page: {e}")
            break  # Exit loop if there's no next page


        # Wait for the next page to load
        time.sleep(3)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Quit the driver once finished
    driver.quit()
