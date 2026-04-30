"""Image transformation helpers."""

from pathlib import Path

import cv2


def normalize_rotation(rotation: int | str | None) -> int:
    try:
        value = int(rotation or 0)
    except (TypeError, ValueError):
        value = 0
    value %= 360
    if value not in (0, 90, 180, 270):
        raise ValueError("rotation must be one of 0, 90, 180, or 270")
    return value


def rotate_image(img, rotation: int):
    rotation = normalize_rotation(rotation)
    if rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def rotated_copy(image_path: Path, output_path: Path, rotation: int) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = rotate_image(img, rotation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path
