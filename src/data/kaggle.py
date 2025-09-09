from typing import Any

import kagglehub
import os

print(os.environ['KAGGLEHUB_CACHE'])

def load_from_kaggle(handle: str) -> Any:
    return kagglehub.dataset_download(handle)
