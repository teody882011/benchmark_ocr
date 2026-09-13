import os
import time
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 1. HELPER FUNCTIONS: Evaluation Metrics (CER & WER)
# ---------------------------------------------------------------------------
def levenshtein_distance(s1, s2):
    """Calculates Levenshtein edit distance between two sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )
    return dp[m][n]

def calculate_cer(reference, hypothesis):
    """Calculates Character Error Rate (CER)."""
    if len(reference) == 0:
        return 0.0
    return levenshtein_distance(reference, hypothesis) / float(len(reference))

def calculate_wer(reference, hypothesis):
    """Calculates Word Error Rate (WER)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if len(ref_words) == 0:
        return 0.0
    return levenshtein_distance(ref_words, hyp_words) / float(len(ref_words))

# ---------------------------------------------------------------------------
# 2. SYNTHETIC DATASET GENERATOR
# ---------------------------------------------------------------------------
def generate_sample_image(text, font_size=8, dpi=72, filename="sample_8pt_72dpi.png"):
    """Generates a low-DPI image with small-font text for benchmarking."""
    # Scale width/height roughly for 72 DPI rendering
    width, height = 800, 150
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Attempt to load a standard sans-serif font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text((10, 20), text, fill=(0, 0, 0), font=font)
    image.save(filename, dpi=(dpi, dpi))
    return filename

# ---------------------------------------------------------------------------
# 3. PREPROCESSING PIPELINES
# ---------------------------------------------------------------------------
def preprocess_image(image_path, method="baseline"):
    """Applies specific image preprocessing pipelines prior to OCR ingestion."""
    img = cv2.imread(image_path)
    
    if method == "baseline":
        return img
    
    elif method == "2x_bicubic":
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        return resized

    elif method == "3x_bicubic":
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        return resized

    elif method == "4x_lanczos":
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width * 4, height * 4), interpolation=cv2.INTER_LANCZOS4)
        return resized

    elif method == "adaptive":
        # Scale 3x Lanczos + Grayscale + Gaussian Blur + Otsu Thresholding
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width * 3, height * 3), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

# ---------------------------------------------------------------------------
# 4. BENCHMARK EXECUTION SUITE
# ---------------------------------------------------------------------------
def run_benchmark():
    ground_truth = "The quick brown fox jumps over the lazy dog near Mt. Diwalwal."
    sample_file = generate_sample_image(ground_truth, font_size=8, dpi=72)
    
    pipelines = ["baseline", "2x_bicubic", "3x_bicubic", "4x_lanczos", "adaptive"]
    
    print("=" * 75)
    print(f"{'Pipeline':<15} | {'CER':<8} | {'WER':<8} | {'Latency (ms)':<12} | {'Recognized Text'}")
    print("=" * 75)
    
    # Configure Tesseract to use LSTM engine mode (OEM 1) and Page Segmentation Mode (PSM 6)
    custom_config = r'--oem 1 --psm 6'
    
    for pipe in pipelines:
        processed_img = preprocess_image(sample_file, method=pipe)
        
        # Measure execution latency
        start_time = time.time()
        ocr_text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
        latency_ms = (time.time() - start_time) * 1000.0
        
        cer = calculate_cer(ground_truth, ocr_text)
        wer = calculate_wer(ground_truth, ocr_text)
        
        print(f"{pipe:<15} | {cer:<8.4f} | {wer:<8.4f} | {latency_ms:<12.2f} | {ocr_text}")
    print("=" * 75)

if __name__ == "__main__":
    run_benchmark()
