import os
import requests
import zipfile
import logging
from tqdm import tqdm
from data.constants import DATASET_OPTIONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)

def extract_zip(zip_path, remove = False):
    """ Extracts a zip file to the same directory as the zip file."""
    _logger.info(f"Extracting zip file: {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            output_path = os.path.dirname(zip_path)
            zip_ref.extractall(output_path)

        _logger.info(f"Extraction complete. Files extracted to: {output_path}")
        if remove:
            os.remove(zip_path)
            _logger.info(f"Removed zip file: {zip_path}")

    except Exception as e:
        _logger.error(f"Error extracting zip file: {zip_path}: {e}")


def download_dataset(dataset_option = 'all', extract = True):
    """ Downloads the dataset(s) specified in the DATASET_OPTIONS constant."""
    # Prepare datasets to download
    if dataset_option == 'all':
        datasets_to_download = DATASET_OPTIONS.keys()
    elif dataset_option not in DATASET_OPTIONS.keys():
        _logger.error(f"Invalid dataset option. Please choose from the following options: {DATASET_OPTIONS.keys()}")
    else:
        datasets_to_download = [dataset_option]

    # Download datasets
    for option in datasets_to_download:
        _logger.info(f"Downloading dataset: {option}")

        # Get the dataset name and path from dataset options
        ds_url = DATASET_OPTIONS[option]['dataset_url']
        ds_path = DATASET_OPTIONS[option]['path']
        ds_zip_path = os.path.join(ds_path, f"{option}.zip")

        # Create directory if it doesn't exist
        is_installed = os.path.exists(ds_path) and len(os.listdir(ds_path)) > 0

        if not is_installed:
            os.makedirs(ds_path, exist_ok = True)
        else:
            _logger.warning(f"Dataset at {ds_path} already exists. Skipping download.")
            
            # Extract only if the zip is still present and extraction is requested
            if extract and os.path.exists(ds_zip_path):
                extract_zip(ds_zip_path, remove = True)
            continue

        # Retrieve and store the dataset (monitoring progress with tqdm)
        response = requests.get(ds_url, stream = True)
        if response.status_code != 200:
            _logger.error(f"Failed to download dataset '{option}' from '{ds_url}'. Status code: {response.status_code}")

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with tqdm(total = total_size, unit = 'B', unit_scale = True, unit_divisor = 1024) as pbar:
            with open(ds_zip_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)

        # Check if the download was complete
        if total_size != 0 and pbar.n != total_size:
            _logger.error("ERROR, something went wrong during the download.")
        else:
            _logger.info(f"Download complete: {ds_zip_path}")

            # Extract the dataset if [extract] option is True
            if extract:
                extract_zip(ds_zip_path, remove = True)

                
