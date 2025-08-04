"""
Robust watermarking implementation using block‑based DCT embedding.

The goal of this module is to provide simple encode and decode functions for
embedding a binary watermark into an image in a way that is resilient to
JPEG compression and cropping.  The algorithm leverages frequency‑domain
embedding and repetition coding, techniques which have been shown to
increase robustness compared to simple spatial‑domain watermarking.  In
particular, transform‑domain watermarking is better suited to survive
operations such as compression, filtering and cropping【52984152619105†L124-L137】.

Key design features:

* **Block‑based DCT:**  The luminance channel of the input image is divided
  into 8×8 blocks.  Each block is transformed into the frequency domain
  using the 2‑D Discrete Cosine Transform (DCT).  A mid‑frequency coefficient
  is modified to encode one watermark bit.  Operating in the frequency
  domain provides improved resilience to compression and common signal
  distortions【544489847968068†L79-L90】.  The DC coefficient (top‑left corner)
  is avoided to maintain image quality.

* **Repetition coding:**  Each watermark bit is repeated `repetition` times
  before embedding.  The extended bitstream is then tiled across all
  available blocks.  During extraction, the decoder collects all votes
  associated with a particular bit and performs majority voting to recover
  the original bits.  Replicating the pattern across the image helps the
  watermark survive cropping【663650263021306†L190-L193】.

* **Simple sign modulation:**  To embed a bit, a fixed strength value
  (`alpha`) is added or subtracted from the chosen DCT coefficient.  A
  positive perturbation encodes a `1` and a negative perturbation encodes a
  `0`.  When extracting, the sign of the coefficient indicates the embedded
  bit.  This approach avoids the need to store the original host image for
  comparison (i.e., it is a blind watermark).

Notes:

* This implementation deliberately keeps the algorithm simple for ease of
  understanding and integration into a student project.  More sophisticated
  schemes might choose embedding positions adaptively based on the human
  visual system, combine DWT/DCT/SVD transforms, or use stronger error
  correcting codes.  For robust protection against tampering, consider
  incorporating Reed‑Solomon codes or chaotic encryption【533662997004930†L70-L88】.

* Because the algorithm modifies mid‑frequency coefficients, the resulting
  watermarked image should retain good perceptual quality; nonetheless,
  users should experiment with the `alpha` parameter to balance
  imperceptibility and robustness.

Usage example:

```
from robust_watermarking import embed_watermark, extract_watermark, bits_to_bytes, bytes_to_bits

# prepare watermark bits (for example, a short message)
message = "HELLO"
wm_bits = bytes_to_bits(message.encode('utf-8'))

# embed the watermark into an image
watermarked_img = embed_watermark("input.jpg", wm_bits, alpha=5, repetition=5)
cv2.imwrite("watermarked.jpg", watermarked_img)

# later: extract the watermark
recovered_bits = extract_watermark(watermarked_img, len(wm_bits), alpha=5, repetition=5)
recovered_message = bytes(bits_to_bytes(recovered_bits)).decode('utf-8', errors='ignore')
print(recovered_message)
```
"""

import math
from typing import Iterable, List, Tuple

import cv2
import numpy as np


def bytes_to_bits(data: bytes) -> List[int]:
    """Convert a byte string into a list of bits (big‑endian within each byte).

    Args:
        data: Byte string to convert.

    Returns:
        List of 0/1 integers representing the bits.
    """
    bits = []
    for byte in data:
        for i in range(8)[::-1]:  # Big‑endian: most significant bit first
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    """Convert an iterable of bits (0/1) back into a byte string.

    Any extra bits that do not fill a full byte are ignored.

    Args:
        bits: Iterable of integer bits.

    Returns:
        Byte string reconstructed from bits.
    """
    bit_list = list(bits)
    if not bit_list:
        return b""
    # Pad to a multiple of 8 bits
    padded_len = (len(bit_list) // 8) * 8
    bytes_out = bytearray()
    for i in range(0, padded_len, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bit_list[i + j] & 1)
        bytes_out.append(byte)
    return bytes(bytes_out)


def _prepare_watermark_bits(bits: List[int], repetition: int) -> List[int]:
    """Apply repetition coding to a list of bits.

    Each bit is repeated `repetition` times consecutively.  For example,
    [1, 0] with repetition=3 becomes [1, 1, 1, 0, 0, 0].

    Args:
        bits: Original list of 0/1 bits.
        repetition: Number of times to repeat each bit.

    Returns:
        Extended bit list.
    """
    extended = []
    for bit in bits:
        extended.extend([bit] * repetition)
    return extended


def embed_watermark(
    image_path: str,
    watermark_bits: List[int],
    alpha: float = 5.0,
    block_size: int = 8,
    repetition: int = 5,
) -> np.ndarray:
    """Embed a binary watermark into an image using block‑based DCT.

    Args:
        image_path: Path to the input image (any format readable by OpenCV).
        watermark_bits: List of 0/1 bits to embed.
        alpha: Strength of the embedding perturbation; larger values improve
            robustness but may degrade image quality.
        block_size: Size of the square block (default 8×8 for JPEG compatibility).
        repetition: Number of times to repeat each watermark bit before
            spreading across blocks.

    Returns:
        Watermarked image as a NumPy array in BGR format.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    # Convert to YCrCb and extract luminance channel
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_channel, cr, cb = cv2.split(ycc)

    # Prepare watermark bits (repetition coding)
    extended_bits = _prepare_watermark_bits(watermark_bits, repetition)
    bit_length = len(extended_bits)

    # Determine number of blocks
    h, w = y_channel.shape
    blocks_y = h // block_size
    blocks_x = w // block_size

    # Copy of Y channel to modify
    y_watermarked = y_channel.copy()

    bit_idx = 0
    # Iterate over blocks in raster order
    for by in range(blocks_y):
        for bx in range(blocks_x):
            # Extract block
            y0 = by * block_size
            x0 = bx * block_size
            block = y_watermarked[y0 : y0 + block_size, x0 : x0 + block_size]
            # Apply DCT
            dct_block = cv2.dct(block)
            # Choose a mid‑frequency coefficient (e.g., position (3,3))
            i, j = 3, 3
            bit = extended_bits[bit_idx % bit_length]
            # Modify coefficient sign based on bit
            if bit == 1:
                dct_block[i, j] += alpha
            else:
                dct_block[i, j] -= alpha
            # Inverse DCT
            idct_block = cv2.idct(dct_block)
            # Put back into Y channel
            y_watermarked[y0 : y0 + block_size, x0 : x0 + block_size] = idct_block
            bit_idx += 1
    
    # Reconstruct YCrCb and convert back to BGR
    watermarked_ycc = cv2.merge([y_watermarked, cr, cb])
    watermarked_img = cv2.cvtColor(watermarked_ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return watermarked_img


def extract_watermark(
    image: np.ndarray,
    watermark_length: int,
    alpha: float = 5.0,
    block_size: int = 8,
    repetition: int = 5,
) -> List[int]:
    """Extract a binary watermark from a possibly attacked image.

    This function assumes the watermark was embedded using `embed_watermark`
    with the same block size, repetition factor and `alpha`.  The recovery is
    based on the sign of a chosen DCT coefficient in each block, aggregated
    through repetition coding and majority voting.

    Args:
        image: Watermarked image as a NumPy array (BGR or grayscale).
        watermark_length: Number of original bits (before repetition coding).
        alpha: Embedding strength used during embedding.
        block_size: Block size used during embedding.
        repetition: Repetition factor used during embedding.

    Returns:
        A list of recovered 0/1 bits of length `watermark_length`.  Note that
        decoding errors may occur if the image has been heavily cropped or
        compressed; using larger `repetition` or `alpha` improves robustness.
    """
    # Convert to YCrCb and extract luminance channel
    if image.ndim == 3 and image.shape[2] == 3:
        ycc = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2YCrCb)
        y_channel = ycc[:, :, 0]
    else:
        y_channel = image.astype(np.float32)
    
    h, w = y_channel.shape
    blocks_y = h // block_size
    blocks_x = w // block_size

    extended_len = watermark_length * repetition
    # Use a list of lists to collect votes for each extended bit position
    votes = [[] for _ in range(extended_len)]

    block_idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0 = by * block_size
            x0 = bx * block_size
            block = y_channel[y0 : y0 + block_size, x0 : x0 + block_size]
            # Apply DCT
            dct_block = cv2.dct(block)
            i, j = 3, 3  # same coefficient used in embedding
            coeff = dct_block[i, j]
            # Decide bit by sign of modified coefficient
            bit_guess = 1 if coeff > 0 else 0
            # Determine which extended bit position this block corresponds to
            pos = block_idx % extended_len
            votes[pos].append(bit_guess)
            block_idx += 1

    # Majority vote for each extended bit
    extended_bits = []
    for v in votes:
        if not v:
            # No votes collected for this position (e.g., block cropped away)
            extended_bits.append(0)
        else:
            ones = sum(v)
            zeros = len(v) - ones
            extended_bits.append(1 if ones >= zeros else 0)

    # Collapse repetition coding
    recovered_bits = []
    for i in range(watermark_length):
        # gather votes for this original bit across repetition positions
        bit_votes = extended_bits[i * repetition : (i + 1) * repetition]
        ones = sum(bit_votes)
        zeros = repetition - ones
        recovered_bits.append(1 if ones >= zeros else 0)
    return recovered_bits


if __name__ == "__main__":
    # # Simple test when run as a script
    # import argparse
    # parser = argparse.ArgumentParser(description="Embed and extract watermarks using block‑based DCT.")
    # parser.add_argument("mode", choices=["encode", "decode"], help="Whether to embed or extract a watermark.")
    # parser.add_argument("image", help="Input image file (for encode) or watermarked image file (for decode).")
    # parser.add_argument("watermark", help="For encode: text to embed; for decode: ignore.")
    # parser.add_argument("output", help="Output image file (for encode) or output text file (for decode).")
    # parser.add_argument("--alpha", type=float, default=5.0, help="Embedding strength.")
    # parser.add_argument("--block_size", type=int, default=8, help="DCT block size.")
    # parser.add_argument("--repetition", type=int, default=5, help="Repetition factor for error‑resilience.")
    # args = parser.parse_args()
    
    # if args.mode == "encode":
    #     wm_bits = bytes_to_bits(args.watermark.encode("utf-8"))
    #     watermarked = embed_watermark(args.image, wm_bits, alpha=args.alpha, block_size=args.block_size, repetition=args.repetition)
    #     cv2.imwrite(args.output, watermarked)
    #     print(f"Watermark embedded and saved to {args.output}")
    # else:
    #     # decode
    #     img = cv2.imread(args.image)
    #     # We need the length of the watermark; read from provided watermark argument as plain text
    #     true_length = len(bytes_to_bits(args.watermark.encode("utf-8")))
    #     recovered_bits = extract_watermark(img, true_length, alpha=args.alpha, block_size=args.block_size, repetition=args.repetition)
    #     recovered_bytes = bits_to_bytes(recovered_bits)
    #     with open(args.output, "wb") as f:
    #         f.write(recovered_bytes)
    #     print(f"Watermark extracted and written to {args.output}")

    
    # prepare watermark bits (for example, a short message)
    message = "HELLO"
    wm_bits = bytes_to_bits(message.encode('utf-8'))

    # embed the watermark into an image
    # watermarked_img = embed_watermark(r"C:\Users\Vin Sen\chihuahua.webp", wm_bits, alpha=5, repetition=5)
    # cv2.imwrite("watermarked.jpg", watermarked_img)

    # later: extract the watermark
    watermarked_img = cv2.imread("wa_watermarked_cropped.jpg")  # Load the watermarked image
    recovered_bits = extract_watermark(watermarked_img, len(wm_bits), alpha=5, repetition=5)
    recovered_message = bytes(bits_to_bytes(recovered_bits)).decode('utf-8', errors='ignore')
    print(recovered_message)