import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

# Logger initialization
logging.basicConfig(level=logging.INFO,)
logger = logging.getLogger(__name__)

# Downloading path
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'data/raw_imgs')
os.makedirs(data_dir, exist_ok=True)

# API initialization
api = KaggleApi()
api.authenticate()

# Dowload action
dataset_name = 'nicholasnisopoli/lisdataset'
api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
logger.info('Dataset Successfullt Downloaded')