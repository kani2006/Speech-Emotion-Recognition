import os
import requests
import zipfile
from tqdm import tqdm
import urllib.parse

# Define paths
base_dir = '/media/victus/ee48b6cb-9d85-46cc-9dff-c15451b93cf0/SER_Project/dataset'
os.makedirs(base_dir, exist_ok=True)

# Download function with progress bar
def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

# Try multiple URLs for a file
def download_with_fallback(urls, destination):
    for url in urls:
        print(f"Trying: {url}")
        if download_file(url, destination):
            return True
    return False

# Download SAVEE
print("Downloading SAVEE...")
savee_dir = os.path.join(base_dir, 'SAVEE')
os.makedirs(savee_dir, exist_ok=True)

savee_files = ['DC', 'JE', 'JK', 'KL']
savee_downloaded = False

for file in savee_files:
    urls = [
        f'https://github.com/Jakobovski/free-speech-datasets/raw/master/surprise/{file}.zip',
        f'https://raw.githubusercontent.com/Jakobovski/free-speech-datasets/master/surprise/{file}.zip',
        f'https://github.com/tyiannak/pyAudioAnalysis/raw/master/data/surprise/{file}.zip',
        f'https://raw.githubusercontent.com/tyiannak/pyAudioAnalysis/master/data/surprise/{file}.zip'
    ]
    
    destination = os.path.join(savee_dir, f'{file}.zip')
    if download_with_fallback(urls, destination):
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(savee_dir)
        os.remove(destination)
        savee_downloaded = True

# Create ALL directory and move all wav files
if savee_downloaded:
    all_dir = os.path.join(savee_dir, 'ALL')
    os.makedirs(all_dir, exist_ok=True)
    for root, dirs, files in os.walk(savee_dir):
        for file in files:
            if file.endswith('.wav'):
                os.rename(os.path.join(root, file), os.path.join(all_dir, file))
    print("SAVEE downloaded successfully!")
else:
    print("Failed to download SAVEE")

# Download EMO-DB
print("\nDownloading EMO-DB...")
emodb_dir = os.path.join(base_dir, 'EMO-DB')
os.makedirs(emodb_dir, exist_ok=True)

emodb_urls = [
    'https://github.com/Jakobovski/free-speech-datasets/raw/master/emodb/emodb.zip',
    'https://raw.githubusercontent.com/Jakobovski/free-speech-datasets/master/emodb/emodb.zip',
    'https://github.com/tyiannak/pyAudioAnalysis/raw/master/data/emodb/emodb.zip',
    'https://raw.githubusercontent.com/tyiannak/pyAudioAnalysis/master/data/emodb/emodb.zip'
]

emodb_zip = os.path.join(emodb_dir, 'emodb.zip')
if download_with_fallback(emodb_urls, emodb_zip):
    with zipfile.ZipFile(emodb_zip, 'r') as zip_ref:
        zip_ref.extractall(emodb_dir)
    os.remove(emodb_zip)
    print("EMO-DB downloaded successfully!")
else:
    print("Failed to download EMO-DB")

print("\nDownload process completed!")
