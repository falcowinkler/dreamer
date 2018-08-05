import requests
import os

dataset_url = "https://github.com/falcowinkler/n-maps-dataset/raw/master/n_maps.tfrecord"


def download_if_not_present(path):
    if not os.path.isfile(path):
        r = requests.get(dataset_url, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
