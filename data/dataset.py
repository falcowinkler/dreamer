import requests
import os

dataset_url = "https://github.com/falcowinkler/n-maps-dataset/raw/master/n_maps.tfrecord"


def download_file(url, output_file):
    r = requests.get(url, stream=True)
    with open(output_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def download_if_not_present(path_to_file):
    if not os.path.isfile(path_to_file):
        download_file(dataset_url, path_to_file)
