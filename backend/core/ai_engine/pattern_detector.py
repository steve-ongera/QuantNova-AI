"""
QuantNova AI — ai_engine/pattern_detector.py
Candlestick pattern detection using rule-based logic + PyTorch CNN.

Two detection layers:
  1. Rule-based engine  — fast, interpretable, uses OHLCV math
  2. CNN visual engine  — slower, handles complex chart formations
                          from screenshot analysis

Detected patterns feed directly into the predictor confidence score.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger('apps')


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Candle:
    open:  float
    high:  float
    low:   float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        return self.range > 0 and (self.body / self.range) < 0.1


@dataclass
class DetectedPattern:
    name: str
    slug: str
    signal: str          # 'buy' | 'sell' | 'neutral'
    confidence: float    # 0.0 – 1.0
    source: str          # 'rule' | 'cnn'
    description: str = ''


# ─────────────────────────────────────────────
# Rule-Based Pattern Engine
# ─────────────────────────────────────────────

class RuleBasedDetector:
    """
    Fast, deterministic candlestick pattern recognition.
    Uses classical TA rules on OHLCV data.
    """

    def detect(self, candles: list[Candle]) -> list[DetectedPattern]:
        """Run all pattern checks. Requires at least 3 candles."""
        if len(candles) < 3:
            return []

        patterns = []
        c  = candles[-1]   # Current
        p1 = candles[-2]   # 1 bar back
        p2 = candles[-3]   # 2 bars back

        # Single-candle patterns
        patterns += self._check_hammer(c)
        patterns += self._check_shooting_star(c)
        patterns += self._check_marubozu(c)
        patterns += self._check_doji(c)
        patterns += self._check_spinning_top(c)

        # Two-candle patterns
        patterns += self._check_engulfing(c, p1)
        patterns += self._check_harami(c, p1)
        patterns += self._check_tweezer(c, p1)
        patterns += self._check_piercing_line(c, p1)
        patterns += self._check_dark_cloud_cover(c, p1)

        # Three-candle patterns
        patterns += self._check_morning_star(c, p1, p2)
        patterns += self._check_evening_star(c, p1, p2)
        patterns += self._check_three_white_soldiers(c, p1, p2)
        patterns += self._check_three_black_crows(c, p1, p2)

        return [p for p in patterns if p is not None]

    # ── Single candle ──────────────────────

    def _check_hammer(self, c: Candle) -> list:
        """Small body at top, long lower wick ≥ 2× body. Bullish reversal."""
        if c.range == 0:
            return []
        if (c.lower_wick >= 2 * c.body and
                c.upper_wick <= 0.1 * c.range and
                c.body > 0):
            conf = min(0.85, 0.5 + (c.lower_wick / c.range) * 0.5)
            return [DetectedPattern(
                name='Hammer', slug='hammer',
                signal='buy', confidence=round(conf, 3),
                source='rule',
                description='Long lower wick indicates buyers stepping in from lows.'
            )]
        return []

    def _check_shooting_star(self, c: Candle) -> list:
        """Small body at bottom, long upper wick ≥ 2× body. Bearish reversal."""
        if c.range == 0:
            return []
        if (c.upper_wick >= 2 * c.body and
                c.lower_wick <= 0.1 * c.range and
                c.body > 0):
            conf = min(0.85, 0.5 + (c.upper_wick / c.range) * 0.5)
            return [DetectedPattern(
                name='Shooting Star', slug='shooting_star',
                signal='sell', confidence=round(conf, 3),
                source='rule',
                description='Long upper wick shows rejection from highs — sellers in control.'
            )]
        return []

    def _check_marubozu(self, c: Candle) -> list:
        """Full-body candle with no significant wicks. Strong momentum."""
        if c.range == 0 or c.body == 0:
            return []
        if c.body / c.range >= 0.92:
            signal = 'buy' if c.is_bullish else 'sell'
            return [DetectedPattern(
                name='Bullish Marubozu' if c.is_bullish else 'Bearish Marubozu',
                slug='bullish_marubozu' if c.is_bullish else 'bearish_marubozu',
                signal=signal, confidence=0.78,
                source='rule',
                description='Strong momentum candle with no rejection — clean directional move.'
            )]
        return []

    def _check_doji(self, c: Candle) -> list:
        """Near-equal open/close. Market indecision."""
        if c.is_doji:
            return [DetectedPattern(
                name='Doji', slug='doji',
                signal='neutral', confidence=0.65,
                source='rule',
                description='Indecision — buyers and sellers balanced. Watch for next candle.'
            )]
        return []

    def _check_spinning_top(self, c: Candle) -> list:
        """Small body with wicks on both sides. Indecision."""
        if c.range == 0:
            return []
        if (c.body / c.range < 0.3 and
                c.upper_wick > 0.1 * c.range and
                c.lower_wick > 0.1 * c.range):
            return [DetectedPattern(
                name='Spinning Top', slug='spinning_top',
                signal='neutral', confidence=0.55,
                source='rule',
                description='Small body with equal wicks — neither side dominating.'
            )]
        return []

    # ── Two-candle ──────────────────────────

    def _check_engulfing(self, c: Candle, p: Candle) -> list:
        """Current candle body fully engulfs previous. Strong reversal signal."""
        if c.body == 0 or p.body == 0:
            return []

        bull_engulf = (c.is_bullish and p.is_bearish and
                       c.open < p.close and c.close > p.open)
        bear_engulf = (c.is_bearish and p.is_bullish and
                       c.open > p.close and c.close < p.open)

        size_ratio = c.body / p.body if p.body > 0 else 1
        conf = min(0.90, 0.65 + min(size_ratio - 1, 0.5) * 0.5)

        if bull_engulf:
            return [DetectedPattern(
                name='Bullish Engulfing', slug='bullish_engulfing',
                signal='buy', confidence=round(conf, 3),
                source='rule',
                description='Bulls completely reversed previous bearish move. Strong buy signal.'
            )]
        if bear_engulf:
            return [DetectedPattern(
                name='Bearish Engulfing', slug='bearish_engulfing',
                signal='sell', confidence=round(conf, 3),
                source='rule',
                description='Bears completely reversed previous bullish move. Strong sell signal.'
            )]
        return []

    def _check_harami(self, c: Candle, p: Candle) -> list:
        """Current candle body contained within previous. Reversal warning."""
        if p.body == 0:
            return []
        p_high = max(p.open, p.close)
        p_low  = min(p.open, p.close)
        c_high = max(c.open, c.close)
        c_low  = min(c.open, c.close)

        if c_high <= p_high and c_low >= p_low and c.body < p.body * 0.5:
            signal = 'buy' if p.is_bearish else 'sell'
            return [DetectedPattern(
                name='Bullish Harami' if signal == 'buy' else 'Bearish Harami',
                slug='bullish_harami' if signal == 'buy' else 'bearish_harami',
                signal=signal, confidence=0.60,
                source='rule',
                description='Inside bar — momentum slowing, potential reversal forming.'
            )]
        return []

    def _check_tweezer(self, c: Candle, p: Candle) -> list:
        """Two candles with matching highs (top) or lows (bottom)."""
        tolerance = (c.range + p.range) * 0.02
        if abs(c.high - p.high) <= tolerance and c.is_bearish and p.is_bullish:
            return [DetectedPattern(
                name='Tweezer Top', slug='tweezer_top',
                signal='sell', confidence=0.68,
                source='rule',
                description='Twin highs — double rejection at resistance. Bearish reversal.'
            )]
        if abs(c.low - p.low) <= tolerance and c.is_bullish and p.is_bearish:
            return [DetectedPattern(
                name='Tweezer Bottom', slug='tweezer_bottom',
                signal='buy', confidence=0.68,
                source='rule',
                description='Twin lows — double rejection at support. Bullish reversal.'
            )]
        return []

    def _check_piercing_line(self, c: Candle, p: Candle) -> list:
        """Bullish: bearish candle followed by candle closing above midpoint."""
        if p.is_bearish and c.is_bullish:
            midpoint = p.open - (p.body / 2)
            if c.open < p.close and c.close > midpoint:
                return [DetectedPattern(
                    name='Piercing Line', slug='piercing_line',
                    signal='buy', confidence=0.70,
                    source='rule',
                    description='Bulls pierced through 50% of previous bearish candle — reversal forming.'
                )]
        return []

    def _check_dark_cloud_cover(self, c: Candle, p: Candle) -> list:
        """Bearish: bullish candle followed by candle closing below midpoint."""
        if p.is_bullish and c.is_bearish:
            midpoint = p.open + (p.body / 2)
            if c.open > p.close and c.close < midpoint:
                return [DetectedPattern(
                    name='Dark Cloud Cover', slug='dark_cloud_cover',
                    signal='sell', confidence=0.70,
                    source='rule',
                    description='Bears pierced through 50% of bullish move — reversal forming.'
                )]
        return []

    # ── Three-candle ────────────────────────

    def _check_morning_star(self, c: Candle, p1: Candle, p2: Candle) -> list:
        """Bearish candle → small body gap down → bullish recovery. Strong buy."""
        if (p2.is_bearish and p2.body > p2.range * 0.5 and
                p1.body < p1.range * 0.3 and
                c.is_bullish and c.close > p2.open - p2.body * 0.5):
            return [DetectedPattern(
                name='Morning Star', slug='morning_star',
                signal='buy', confidence=0.82,
                source='rule',
                description='Three-candle bullish reversal: exhaustion → indecision → bull takeover.'
            )]
        return []

    def _check_evening_star(self, c: Candle, p1: Candle, p2: Candle) -> list:
        """Bullish candle → small body gap up → bearish reversal. Strong sell."""
        if (p2.is_bullish and p2.body > p2.range * 0.5 and
                p1.body < p1.range * 0.3 and
                c.is_bearish and c.close < p2.open + p2.body * 0.5):
            return [DetectedPattern(
                name='Evening Star', slug='evening_star',
                signal='sell', confidence=0.82,
                source='rule',
                description='Three-candle bearish reversal: exhaustion → indecision → bear takeover.'
            )]
        return []

    def _check_three_white_soldiers(self, c: Candle, p1: Candle, p2: Candle) -> list:
        """Three consecutive strong bullish candles. Powerful uptrend confirmation."""
        if (c.is_bullish and p1.is_bullish and p2.is_bullish and
                c.close > p1.close > p2.close and
                c.body > c.range * 0.6 and
                p1.body > p1.range * 0.6 and
                p2.body > p2.range * 0.6):
            return [DetectedPattern(
                name='Three White Soldiers', slug='three_white_soldiers',
                signal='buy', confidence=0.85,
                source='rule',
                description='Three consecutive strong bullish candles — sustained buying pressure.'
            )]
        return []

    def _check_three_black_crows(self, c: Candle, p1: Candle, p2: Candle) -> list:
        """Three consecutive strong bearish candles. Powerful downtrend confirmation."""
        if (c.is_bearish and p1.is_bearish and p2.is_bearish and
                c.close < p1.close < p2.close and
                c.body > c.range * 0.6 and
                p1.body > p1.range * 0.6 and
                p2.body > p2.range * 0.6):
            return [DetectedPattern(
                name='Three Black Crows', slug='three_black_crows',
                signal='sell', confidence=0.85,
                source='rule',
                description='Three consecutive strong bearish candles — sustained selling pressure.'
            )]
        return []


# ─────────────────────────────────────────────
# CNN Visual Pattern Detector
# ─────────────────────────────────────────────

class CNNPatternDetector:
    """
    PyTorch CNN for detecting complex chart formations from screenshots.
    Handles patterns that require visual context beyond OHLCV math:
      - Head and Shoulders
      - Double Top / Double Bottom
      - Rising / Falling Wedge
      - Bull / Bear Flag
      - Triangle formations
      - Supply & Demand zones
    """

    PATTERN_LABELS = [
        'head_and_shoulders', 'inverse_head_and_shoulders',
        'double_top', 'double_bottom',
        'rising_wedge', 'falling_wedge',
        'bull_flag', 'bear_flag',
        'ascending_triangle', 'descending_triangle', 'symmetrical_triangle',
        'supply_zone', 'demand_zone',
        'breakout_up', 'breakout_down',
        'none',
    ]

    PATTERN_SIGNALS = {
        'head_and_shoulders': 'sell',
        'inverse_head_and_shoulders': 'buy',
        'double_top': 'sell',
        'double_bottom': 'buy',
        'rising_wedge': 'sell',
        'falling_wedge': 'buy',
        'bull_flag': 'buy',
        'bear_flag': 'sell',
        'ascending_triangle': 'buy',
        'descending_triangle': 'sell',
        'symmetrical_triangle': 'neutral',
        'supply_zone': 'sell',
        'demand_zone': 'buy',
        'breakout_up': 'buy',
        'breakout_down': 'sell',
        'none': 'neutral',
    }

    PATTERN_DESCRIPTIONS = {
        'head_and_shoulders': 'Classic bearish reversal — left shoulder, higher head, right shoulder.',
        'inverse_head_and_shoulders': 'Classic bullish reversal — inverse H&S at lows.',
        'double_top': 'Two equal highs — price failed twice at resistance. Bearish.',
        'double_bottom': 'Two equal lows — price bounced twice from support. Bullish.',
        'rising_wedge': 'Converging upward lines — buyers losing momentum. Bearish.',
        'falling_wedge': 'Converging downward lines — sellers losing momentum. Bullish.',
        'bull_flag': 'Sharp rally followed by orderly pullback — continuation up.',
        'bear_flag': 'Sharp drop followed by weak bounce — continuation down.',
        'ascending_triangle': 'Flat resistance + rising support — buyers accumulating.',
        'descending_triangle': 'Flat support + falling resistance — sellers distributing.',
        'symmetrical_triangle': 'Coiling price — breakout direction determines bias.',
        'supply_zone': 'Area of strong selling — price likely to reverse here.',
        'demand_zone': 'Area of strong buying — price likely to bounce here.',
        'breakout_up': 'Price breaking above key level with momentum.',
        'breakout_down': 'Price breaking below key level with momentum.',
        'none': 'No clear chart pattern identified.',
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._device = None
        self._load_model()

    def _load_model(self):
        """Load PyTorch pattern detection model."""
        import importlib
        if not self.model_path:
            return
        try:
            import torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model = torch.load(self.model_path, map_location=self._device)
            self._model.eval()
            logger.info(f"[CNNPattern] Model loaded on {self._device}")
        except Exception as e:
            logger.warning(f"[CNNPattern] Could not load model: {e}")
            self._model = None

    def detect(self, cnn_features: Optional[np.ndarray]) -> list[DetectedPattern]:
        """
        Run CNN pattern detection on feature vector from ImageAnalyzer.
        Returns detected chart patterns with confidence scores.
        """
        if self._model is None or cnn_features is None:
            return self._fallback_detection(cnn_features)

        try:
            import torch
            tensor = torch.tensor(cnn_features, dtype=torch.float32).unsqueeze(0).to(self._device)
            with torch.no_grad():
                logits = self._model(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

            patterns = []
            for idx, prob in enumerate(probs):
                if prob >= 0.40 and idx < len(self.PATTERN_LABELS):
                    label = self.PATTERN_LABELS[idx]
                    if label == 'none':
                        continue
                    patterns.append(DetectedPattern(
                        name=label.replace('_', ' ').title(),
                        slug=label,
                        signal=self.PATTERN_SIGNALS.get(label, 'neutral'),
                        confidence=round(float(prob), 3),
                        source='cnn',
                        description=self.PATTERN_DESCRIPTIONS.get(label, ''),
                    ))

            patterns.sort(key=lambda p: p.confidence, reverse=True)
            return patterns[:3]  # Return top 3 chart patterns

        except Exception as e:
            logger.error(f"[CNNPattern] Inference error: {e}", exc_info=True)
            return []

    def _fallback_detection(self, features: Optional[np.ndarray]) -> list[DetectedPattern]:
        """
        When no model is loaded, use feature vector heuristics
        to make a rough pattern guess (for development/testing).
        """
        if features is None:
            return []
        # Use mean of feature vector as a simple heuristic
        mean_val = float(np.mean(features))
        if mean_val > 0.6:
            return [DetectedPattern(
                name='Demand Zone', slug='demand_zone',
                signal='buy', confidence=0.52,
                source='cnn',
                description='Possible demand zone detected (heuristic mode — no model loaded).'
            )]
        elif mean_val < 0.3:
            return [DetectedPattern(
                name='Supply Zone', slug='supply_zone',
                signal='sell', confidence=0.52,
                source='cnn',
                description='Possible supply zone detected (heuristic mode — no model loaded).'
            )]
        return []


# ─────────────────────────────────────────────
# Combined Pattern Detector (Public API)
# ─────────────────────────────────────────────

class PatternDetector:
    """
    Unified interface combining rule-based + CNN pattern detection.
    Usage:
        detector = PatternDetector(cnn_model_path='models_trained/pattern_detector.pt')
        patterns = detector.detect(candles=candles, cnn_features=features)
    """

    def __init__(self, cnn_model_path: Optional[str] = None):
        self.rule_detector = RuleBasedDetector()
        self.cnn_detector  = CNNPatternDetector(model_path=cnn_model_path)

    def detect(
        self,
        candles: Optional[list] = None,
        cnn_features: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """
        Run both detection engines and merge results.

        Args:
            candles:      list of Candle objects (for rule-based detection)
            cnn_features: CNN feature vector from ImageAnalyzer (for visual detection)

        Returns:
            Sorted list of pattern dicts with merged confidence scores.
        """
        all_patterns: list[DetectedPattern] = []

        # Rule-based detection (OHLCV)
        if candles:
            candle_objs = [
                Candle(**c) if isinstance(c, dict) else c
                for c in candles
            ]
            all_patterns += self.rule_detector.detect(candle_objs)

        # CNN visual detection (chart image)
        if cnn_features is not None:
            all_patterns += self.cnn_detector.detect(cnn_features)

        # Deduplicate by slug (take highest confidence)
        seen = {}
        for p in all_patterns:
            if p.slug not in seen or p.confidence > seen[p.slug].confidence:
                seen[p.slug] = p

        merged = sorted(seen.values(), key=lambda p: p.confidence, reverse=True)

        logger.info(
            f"[PatternDetector] Detected {len(merged)} patterns: "
            f"{[p.slug for p in merged[:5]]}"
        )

        return [
            {
                'name': p.name,
                'slug': p.slug,
                'signal': p.signal,
                'confidence': p.confidence,
                'source': p.source,
                'description': p.description,
            }
            for p in merged
        ]

    def aggregate_signal(self, patterns: list[dict]) -> tuple[str, float]:
        """
        Aggregate all detected patterns into a single signal + confidence.

        Returns:
            (signal, confidence) — e.g. ('buy', 0.74)
        """
        if not patterns:
            return ('hold', 0.40)

        buy_score  = sum(p['confidence'] for p in patterns if p['signal'] == 'buy')
        sell_score = sum(p['confidence'] for p in patterns if p['signal'] == 'sell')
        total      = buy_score + sell_score

        if total == 0:
            return ('hold', 0.40)

        if buy_score > sell_score:
            conf = buy_score / total
            return ('buy', round(min(conf, 0.95), 3))
        elif sell_score > buy_score:
            conf = sell_score / total
            return ('sell', round(min(conf, 0.95), 3))
        else:
            return ('hold', 0.50)