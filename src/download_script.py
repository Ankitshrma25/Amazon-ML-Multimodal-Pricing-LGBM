import pandas as pd
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
# Use the paths specific to your project structure
DATA_PATH = 'student_resource/dataset/train.csv'
IMAGE_DIR = 'student_resource/images'
MAX_THREADS = 10  # REDUCED THREAD COUNT to safely bypass the Windows '63 handles' limit
MAX_RETRIES = 3
TIMEOUT = 15 # Timeout in seconds

# --- Setup ---
os.makedirs(IMAGE_DIR, exist_ok=True)
df_train = pd.read_csv(DATA_PATH)

def download_image(row):
    """Downloads a single image with retry logic."""
    url = row['image_link']
    # Use sample_id as the filename (e.g., 33127.jpg)
    filename = os.path.join(IMAGE_DIR, f"{row['sample_id']}.jpg")
    
    # 1. Skip if already downloaded
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return 

    # 2. Skip if the link is missing or NaN
    if pd.isna(url):
        # Create a tiny file indicating failure to skip it later
        with open(filename, 'w') as f:
            f.write("LINK_MISSING")
        return

    # 3. Download with Retries
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            
            if response.status_code == 200 and response.content:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return
            
        except requests.exceptions.RequestException:
            # Sleep briefly before retrying a connection error
            import time; time.sleep(1)
            continue 
            
    # 4. If all retries fail, mark as failed
    with open(filename, 'w') as f:
        f.write("DOWNLOAD_FAILED")


if __name__ == '__main__':
    print(f"Starting download of {len(df_train)} images...")
    
    # Use ThreadPoolExecutor to download images concurrently for speed
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Get the iterator of rows from the DataFrame
        rows_to_process = [row for index, row in df_train.iterrows()]
        
        # Process rows and show progress with tqdm
        list(tqdm(executor.map(download_image, rows_to_process), 
                  total=len(df_train), 
                  desc="Downloading Images"))
        
    print(f"\nImage download complete. Check the '{IMAGE_DIR}' folder.")