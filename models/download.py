import os
import requests
from tqdm import tqdm


def download_file(url, dest):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(dest, "wb") as file, tqdm(
        desc=dest,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:

        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)