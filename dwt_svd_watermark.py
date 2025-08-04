"""
DWT + SVD image watermarking (Python 3)

Dependencies
------------
numpy, opencv-python, PyWavelets (pywt)

Install missing packages:
    pip install numpy opencv-python pywavelets

Usage
-----
    import dwt_svd_watermark as wm

    # Embed
    watermarked = wm.embed_watermark('cover.jpg', 'logo.png',
                                     alpha=0.04, level=2, wavelet='haar')
    cv2.imwrite('watermarked.jpg', watermarked)

    # Extract (using original cover image)
    extracted = wm.extract_watermark('watermarked.jpg', 'cover.jpg',
                                     watermark_shape=(64, 64),
                                     alpha=0.04, level=2, wavelet='haar')
    cv2.imwrite('extracted_logo.png', extracted*255)

Notes
-----
* This implementation is *semi‑blind*: extraction uses the cover image.
* Robustness comes from:
      – Frequency‑domain embedding (2‑level DWT)
      – Inserting the mark into singular values of LH2 (mid‑frequency) sub‑band
      – Replicating (tiling) the watermark so many copies survive cropping
      – Scaling factor `alpha` controls imperceptibility vs robustness
* For full‑blind extraction, embed synchronization patterns and/or store
  original singular values as a key.
"""

import cv2
import numpy as np
import pywt


def _prepare_watermark(wm_path, target_shape):
    wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    # Fix: Handle 1D target_shape from SVD
    if len(target_shape) == 1:
        # For SVD singular values, create a square shape
        size = int(np.sqrt(target_shape[0]))
        target_shape = (size, size)
    
    wm = cv2.resize(wm, target_shape, interpolation=cv2.INTER_AREA)
    wm = (wm > 127).astype(np.float32)
    return wm


def embed_watermark(cover_path: str,
                    watermark_path: str,
                    alpha: float = 0.04,
                    level: int = 2,
                    wavelet: str = 'haar'):
    """Return a watermarked BGR image as uint8."""
    cover = cv2.imread(cover_path)
    if cover is None:
        raise FileNotFoundError(cover_path)
    # work in Y channel for less visible artifacts
    ycbcr = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycbcr[:, :, 0]

    # multi‑level DWT
    coeffs = pywt.wavedec2(Y, wavelet=wavelet, level=level)
    # coeffs[1:] is list of (LH, HL, HH) tuples
    (LH2, HL2, HH2) = coeffs[1]

    # Use LH2 (or choose any) for embedding
    subband = LH2.astype(np.float32)

    # SVD
    U, S, Vt = np.linalg.svd(subband, full_matrices=False)

    # prepare/tile watermark to size of S
    wm = _prepare_watermark(watermark_path, S.shape)
    
    # Flatten the watermark and tile/repeat to match S length
    wm_flat = wm.flatten()
    watermark_tiled = np.tile(wm_flat, (len(S) // len(wm_flat)) + 1)[:len(S)]

    # embed
    S_emb = S + alpha * watermark_tiled

    # inverse SVD
    subband_emb = (U @ np.diag(S_emb) @ Vt)

    # put back
    coeffs_emb = list(coeffs)
    coeffs_emb[1] = (subband_emb, HL2, HH2)

    # inverse DWT
    Y_emb = pywt.waverec2(coeffs_emb, wavelet=wavelet)

    # clip and merge
    ycbcr[:, :, 0] = np.clip(Y_emb, 0, 255)
    watermarked = cv2.cvtColor(ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return watermarked


def extract_watermark(watermarked_path: str,
                      cover_path: str,
                      watermark_shape: tuple[int, int],
                      alpha: float = 0.04,
                      level: int = 2,
                      wavelet: str = 'haar'):
    """Return extracted watermark in float range [0,1]."""
    wmk = cv2.imread(watermarked_path)
    cover = cv2.imread(cover_path)
    if wmk is None or cover is None:
        raise FileNotFoundError()

    def _get_subband(img):
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        coeffs = pywt.wavedec2(y, wavelet=wavelet, level=level)
        subband, _, _ = coeffs[1]  # LH2
        return subband

    Sb_wmk = _get_subband(wmk)
    Sb_orig = _get_subband(cover)

    # SVD on both
    _, S_w, _ = np.linalg.svd(Sb_wmk, full_matrices=False)
    _, S_o, _ = np.linalg.svd(Sb_orig, full_matrices=False)

    # difference
    W_est = (S_w - S_o) / alpha
    W_est = np.clip(W_est, 0, 1)

    # Reshape to 2D before resize
    size = int(np.sqrt(len(W_est)))
    W_est_2d = W_est[:size*size].reshape(size, size)
    
    # Resize to original watermark_shape
    extracted = cv2.resize(W_est_2d, watermark_shape, interpolation=cv2.INTER_NEAREST)
    return extracted

if __name__ == "__main__":
    # Embed
    # watermarked = embed_watermark(r"C:\Users\Vin Sen\chihuahua.webp", 'copper.jpeg', alpha=0.04, level=2, wavelet='haar')
    # cv2.imwrite('watermarked_copper.jpg', watermarked)

    # Extract (using original cover image)
    extracted = extract_watermark('watermarked_copper.jpg', r"C:\Users\Vin Sen\chihuahua.webp", watermark_shape=(64, 64), alpha=0.04, level=2, wavelet='haar')
    cv2.imwrite('extracted_watermark.jpg', (extracted * 255).astype(np.uint8))

    # cv2.imshow('wm', extracted)  
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()