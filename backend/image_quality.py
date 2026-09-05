import cv2
import numpy as np


def _focus_score(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _illumination_score(gray: np.ndarray, tiles: int = 4) -> float:
    h, w = gray.shape
    th, tw = h // tiles, w // tiles
    means = []
    for i in range(tiles):
        for j in range(tiles):
            tile = gray[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            if tile.size:
                means.append(tile.mean())
    return float(np.std(means))


def _fov_coverage(gray: np.ndarray) -> float:
    mask = (gray > 15).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest) / (gray.shape[0] * gray.shape[1])


def assess_quality(img_rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    focus = _focus_score(gray)
    illum = _illumination_score(gray)
    fov = _fov_coverage(gray)

    reasons = []
    if focus < 80:
        reasons.append("image too blurry - refocus and recapture")
    if fov < 0.75:
        reasons.append("retina not centered - recapture with full field of view")
    if illum > 35:
        reasons.append("uneven illumination - check flash and positioning")

    if focus < 40 or fov < 0.55:
        decision = "reject"
    elif reasons:
        decision = "borderline"
    else:
        decision = "gradable"

    return {
        "focus_score": float(focus),
        "illum_score": float(illum),
        "fov_coverage": float(fov),
        "decision": decision,
        "feedback": "; ".join(reasons) if reasons else "",
    }


def enhance_image(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=30)
    bg_channel_mean = bg.mean(axis=(0, 1), keepdims=True)
    corrected = cv2.normalize(img - bg + bg_channel_mean, None, 0, 255, cv2.NORM_MINMAX)
    corrected = corrected.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    r, g, b = cv2.split(corrected)
    g_eq = clahe.apply(g)
    merged = cv2.merge([r, g_eq, b])

    enhanced = cv2.fastNlMeansDenoisingColored(
        merged, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
    )
    return enhanced
