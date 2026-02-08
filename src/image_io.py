import os
from typing import Optional, Tuple

import cv2
import numpy as np


def read_image_rgb(path: str, min_size: int = 64) -> Tuple[Optional[np.ndarray], str]:
    if not os.path.exists(path):
        return None, "missing"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, "corrupt"
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if height < min_size or width < min_size:
        return None, "too_small"
    return image, "ok"
