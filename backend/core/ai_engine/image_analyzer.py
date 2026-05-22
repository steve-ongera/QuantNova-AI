"""
QuantNova AI — ai_engine/image_analyzer.py
OpenCV preprocessing + CNN feature extraction for forex chart images.

Responsibilities:
  - Load and validate chart images
  - Preprocess: resize, normalize, denoise, enhance contrast
  - Extract visual features: edges, zones, trend lines
  - Feed prepared tensor to CNN model
  - Return raw feature map + detected visual regions
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger('apps')

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CNN_INPUT_SIZE = (224, 224)       # Standard input for MobileNetV2 / ResNet50
NORMALIZE_MEAN = [0.485, 0.456, 0.406]   # ImageNet means
NORMALIZE_STD  = [0.229, 0.224, 0.225]   # ImageNet stds

# Candlestick body color detection ranges (HSV)
BULLISH_GREEN_HSV = [(40, 40, 40), (80, 255, 255)]
BEARISH_RED_HSV   = [(0, 40, 40), (15, 255, 255)]


class ImageAnalyzer:
    """
    Handles all image preprocessing and low-level visual feature extraction
    before passing data to the CNN model and pattern detector.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._load_model()

    # ─────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the CNN model (TensorFlow/Keras .h5)."""
        if not self.model_path or not Path(self.model_path).exists():
            logger.warning("[ImageAnalyzer] No CNN model file found — running in feature-only mode.")
            self._model = None
            return
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self.model_path)
            logger.info(f"[ImageAnalyzer] CNN model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"[ImageAnalyzer] Failed to load CNN model: {e}")
            self._model = None

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def analyze(self, image_path: str) -> dict:
        """
        Full analysis pipeline for a single chart image.

        Returns:
            {
                'preprocessed': np.ndarray,          # (224,224,3) normalized tensor
                'cnn_features': np.ndarray | None,   # CNN feature vector
                'visual': {
                    'candle_ratio': float,            # bullish/bearish candle ratio
                    'trend_direction': str,           # 'up' | 'down' | 'sideways'
                    'trend_angle': float,             # degrees
                    'volume_profile': list,           # relative volume bars
                    'price_zones': list,              # detected price zone boxes
                    'wick_dominance': str,            # 'upper' | 'lower' | 'balanced'
                    'chart_quality': float,           # 0.0–1.0 image clarity score
                }
            }
        """
        raw = self._load_image(image_path)
        if raw is None:
            raise ValueError(f"Could not load image: {image_path}")

        preprocessed = self._preprocess(raw)
        visual = self._extract_visual_features(raw)
        cnn_features = self._run_cnn(preprocessed) if self._model else None

        logger.debug(
            f"[ImageAnalyzer] trend={visual['trend_direction']} "
            f"candle_ratio={visual['candle_ratio']:.2f} "
            f"quality={visual['chart_quality']:.2f}"
        )

        return {
            'preprocessed': preprocessed,
            'cnn_features': cnn_features,
            'visual': visual,
        }

    # ─────────────────────────────────────────
    # Image Loading
    # ─────────────────────────────────────────

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from disk. Returns BGR numpy array or None."""
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"[ImageAnalyzer] cv2.imread failed for: {image_path}")
        return img

    # ─────────────────────────────────────────
    # Preprocessing Pipeline
    # ─────────────────────────────────────────

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. Resize to CNN input size
        2. Convert BGR → RGB
        3. Denoise (mild bilateral filter)
        4. Enhance contrast (CLAHE on L channel)
        5. Normalize with ImageNet stats
        Returns float32 array (224, 224, 3).
        """
        # Resize
        resized = cv2.resize(img, CNN_INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)

        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Mild bilateral denoise (preserve edges = candlestick borders)
        denoised = cv2.bilateralFilter(rgb, d=5, sigmaColor=15, sigmaSpace=15)

        # CLAHE contrast enhancement on L channel
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)

        # Normalize to float32
        tensor = enhanced.astype(np.float32) / 255.0
        mean = np.array(NORMALIZE_MEAN, dtype=np.float32)
        std  = np.array(NORMALIZE_STD,  dtype=np.float32)
        tensor = (tensor - mean) / std

        return tensor

    # ─────────────────────────────────────────
    # Visual Feature Extraction
    # ─────────────────────────────────────────

    def _extract_visual_features(self, img: np.ndarray) -> dict:
        """Extract human-interpretable visual features from raw BGR image."""
        return {
            'candle_ratio':    self._detect_candle_ratio(img),
            'trend_direction': self._detect_trend_direction(img),
            'trend_angle':     self._detect_trend_angle(img),
            'volume_profile':  self._extract_volume_profile(img),
            'price_zones':     self._detect_price_zones(img),
            'wick_dominance':  self._detect_wick_dominance(img),
            'chart_quality':   self._score_chart_quality(img),
        }

    def _detect_candle_ratio(self, img: np.ndarray) -> float:
        """
        Estimate bullish vs bearish candle ratio using color detection.
        Returns value 0.0 (all bearish) → 1.0 (all bullish).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv,
            np.array(BULLISH_GREEN_HSV[0]),
            np.array(BULLISH_GREEN_HSV[1])
        )
        red_mask = cv2.inRange(hsv,
            np.array(BEARISH_RED_HSV[0]),
            np.array(BEARISH_RED_HSV[1])
        )

        green_pixels = np.count_nonzero(green_mask)
        red_pixels   = np.count_nonzero(red_mask)
        total = green_pixels + red_pixels

        if total == 0:
            return 0.5  # Cannot determine — neutral
        return green_pixels / total

    def _detect_trend_direction(self, img: np.ndarray) -> str:
        """
        Detect overall trend direction using edge detection + Hough lines.
        Analyzes dominant line angles in the price area.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=40, minLineLength=30, maxLineGap=10
        )

        if lines is None:
            return 'sideways'

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)

        if not angles:
            return 'sideways'

        avg_angle = np.median(angles)

        # Note: in image coordinates Y increases downward
        # Negative angle = price going up (line sloping up-right)
        if avg_angle < -10:
            return 'up'
        elif avg_angle > 10:
            return 'down'
        else:
            return 'sideways'

    def _detect_trend_angle(self, img: np.ndarray) -> float:
        """Returns the median trend line angle in degrees."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = [
            np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
            for l in lines if l[0][2] != l[0][0]
        ]
        return float(np.median(angles)) if angles else 0.0

    def _extract_volume_profile(self, img: np.ndarray) -> list:
        """
        Approximate volume profile by analyzing pixel density in
        horizontal bands of the lower 20% of the chart (typical volume area).
        Returns 10 relative volume bars (0.0–1.0).
        """
        h, w = img.shape[:2]
        volume_area = img[int(h * 0.80):, :]  # bottom 20%
        gray = cv2.cvtColor(volume_area, cv2.COLOR_BGR2GRAY)

        num_bars = 10
        bar_w = w // num_bars
        bars = []
        for i in range(num_bars):
            segment = gray[:, i * bar_w:(i + 1) * bar_w]
            # Non-background (dark) pixel density = proxy for volume bar height
            dark_pixels = np.sum(segment < 200)
            bars.append(int(dark_pixels))

        max_bar = max(bars) if max(bars) > 0 else 1
        return [round(b / max_bar, 3) for b in bars]

    def _detect_price_zones(self, img: np.ndarray) -> list:
        """
        Detect horizontal price zones (support/resistance) as bounding boxes.
        Uses horizontal line detection from Hough transform.
        Returns list of dicts: [{'y_pct': float, 'strength': float}]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=60, minLineLength=img.shape[1] * 0.3,
            maxLineGap=20
        )

        if lines is None:
            return []

        h = img.shape[0]
        zones = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Only near-horizontal lines (angle < 5 degrees)
            if abs(y2 - y1) < 8:
                y_pct = round((y1 / h), 3)
                length = abs(x2 - x1)
                strength = min(1.0, length / img.shape[1])
                zones.append({'y_pct': y_pct, 'strength': round(strength, 3)})

        # Deduplicate zones close together (within 5% of height)
        zones.sort(key=lambda z: z['y_pct'])
        deduped = []
        for zone in zones:
            if not deduped or abs(zone['y_pct'] - deduped[-1]['y_pct']) > 0.05:
                deduped.append(zone)

        return deduped[:10]  # Return top 10 zones

    def _detect_wick_dominance(self, img: np.ndarray) -> str:
        """
        Detect whether upper or lower wicks dominate the chart.
        Proxy: compare brightness distribution in top vs bottom halves.
        Heavy upper wicks → rejection from high (bearish pressure).
        Heavy lower wicks → rejection from low (bullish pressure).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        top_half    = gray[:h // 2, :]
        bottom_half = gray[h // 2:, :]

        # Count thin line pixels (wicks = thin vertical dark lines)
        top_thin    = np.sum(top_half < 80)
        bottom_thin = np.sum(bottom_half < 80)

        if top_thin > bottom_thin * 1.3:
            return 'upper'
        elif bottom_thin > top_thin * 1.3:
            return 'lower'
        return 'balanced'

    def _score_chart_quality(self, img: np.ndarray) -> float:
        """
        Score image quality for reliability (0.0 = unusable, 1.0 = perfect).
        Checks: resolution, blur level, contrast ratio.
        """
        h, w = img.shape[:2]
        scores = []

        # Resolution check (at least 400x300 for good analysis)
        res_score = min(1.0, (h * w) / (400 * 300))
        scores.append(res_score)

        # Blur check using Laplacian variance (higher = sharper)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500)
        scores.append(blur_score)

        # Contrast check (std dev of pixel values)
        contrast_score = min(1.0, float(np.std(gray)) / 80)
        scores.append(contrast_score)

        return round(float(np.mean(scores)), 3)

    # ─────────────────────────────────────────
    # CNN Inference
    # ─────────────────────────────────────────

    def _run_cnn(self, tensor: np.ndarray) -> np.ndarray:
        """
        Run preprocessed image through CNN model.
        Returns feature vector (class probabilities or embeddings).
        """
        import tensorflow as tf
        batch = np.expand_dims(tensor, axis=0)  # (1, 224, 224, 3)
        features = self._model.predict(batch, verbose=0)
        return features[0]  # Remove batch dim → (num_classes,)