"""
QuantNova AI — ai_engine/predictor.py
Main prediction engine — orchestrates all AI components into a final signal.

Pipeline:
  ImageAnalyzer → PatternDetector → Signal Fusion → Risk Calculator → Output

The ForexPredictor is the single entry point called by the Celery task.
It combines:
  - CNN visual features
  - Rule-based candlestick patterns
  - Trend analysis from OpenCV
  - Strategy rules from PDF knowledge base
  - Risk/reward calculation
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
from django.conf import settings

from .image_analyzer import ImageAnalyzer
from .pattern_detector import PatternDetector

logger = logging.getLogger('apps')

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

AI_CONFIG = getattr(settings, 'AI_ENGINE', {})

DEFAULT_RISK_REWARD = 2.0       # Minimum acceptable R:R
CONFIDENCE_FLOOR    = 0.40      # Below this = HOLD regardless of signal
CONFIDENCE_CEILING  = 0.95      # Cap to avoid overconfidence

# Trend-to-signal alignment bonus
TREND_ALIGNMENT_BONUS = 0.08

# Timeframe weights (higher TF = more reliable signal)
TIMEFRAME_WEIGHTS = {
    'M1': 0.55, 'M5': 0.62, 'M15': 0.68,
    'M30': 0.72, 'H1': 0.78, 'H4': 0.84,
    'D1': 0.90, 'W1': 0.93, 'MN': 0.95,
}


# ─────────────────────────────────────────────
# Signal Fusion Engine
# ─────────────────────────────────────────────

class SignalFusion:
    """
    Combines evidence from multiple AI components into a single
    Buy / Sell / Hold signal with calibrated confidence.

    Fusion inputs:
      1. Pattern signal + confidence (from PatternDetector)
      2. Trend direction (from ImageAnalyzer visual features)
      3. CNN class probabilities (from ImageAnalyzer CNN)
      4. Candle ratio (bullish/bearish pixel balance)
      5. Timeframe reliability weight
    """

    def fuse(
        self,
        pattern_signal: str,
        pattern_confidence: float,
        trend_direction: str,
        candle_ratio: float,
        cnn_probs: Optional[np.ndarray],
        timeframe: str,
        active_rules: list[dict],
    ) -> dict:
        """
        Fuse all inputs into a final signal.

        Returns:
            {
                'signal': 'buy' | 'sell' | 'hold',
                'confidence': float,
                'components': dict,   # score breakdown for transparency
            }
        """
        components = {}

        # ── 1. Pattern score (highest weight)
        pattern_score = self._signal_to_score(pattern_signal) * pattern_confidence
        components['pattern'] = round(pattern_score, 3)

        # ── 2. Trend alignment bonus
        trend_score = self._trend_score(trend_direction, pattern_signal)
        components['trend'] = round(trend_score, 3)

        # ── 3. Candle ratio score
        # candle_ratio: 0.0 = all bearish, 1.0 = all bullish
        candle_score = (candle_ratio - 0.5) * 0.3   # -0.15 to +0.15
        components['candle_ratio'] = round(candle_score, 3)

        # ── 4. CNN probability score (if available)
        cnn_score = self._cnn_score(cnn_probs) if cnn_probs is not None else 0.0
        components['cnn'] = round(cnn_score, 3)

        # ── 5. Strategy rules alignment
        rule_score = self._rule_alignment_score(active_rules, pattern_signal)
        components['rules'] = round(rule_score, 3)

        # ── Weighted fusion
        raw_score = (
            pattern_score * 0.40 +
            trend_score   * 0.20 +
            cnn_score     * 0.20 +
            candle_score  * 0.10 +
            rule_score    * 0.10
        )

        # ── Timeframe reliability weight
        tf_weight = TIMEFRAME_WEIGHTS.get(timeframe, 0.75)
        adjusted_score = raw_score * tf_weight
        components['timeframe_weight'] = tf_weight

        # ── Determine final signal direction
        if adjusted_score > 0.12:
            signal = 'buy'
            confidence = min(CONFIDENCE_CEILING, 0.50 + adjusted_score)
        elif adjusted_score < -0.12:
            signal = 'sell'
            confidence = min(CONFIDENCE_CEILING, 0.50 + abs(adjusted_score))
        else:
            signal = 'hold'
            confidence = max(0.40, 0.50 - abs(adjusted_score))

        # ── Apply confidence floor
        if confidence < CONFIDENCE_FLOOR:
            signal = 'hold'
            confidence = CONFIDENCE_FLOOR

        return {
            'signal': signal,
            'confidence': round(confidence, 4),
            'components': components,
        }

    def _signal_to_score(self, signal: str) -> float:
        return {'buy': 1.0, 'sell': -1.0, 'neutral': 0.0, 'hold': 0.0}.get(signal, 0.0)

    def _trend_score(self, trend: str, pattern_signal: str) -> float:
        """Reward trend-pattern alignment, penalize divergence."""
        alignment = {
            ('up', 'buy'): TREND_ALIGNMENT_BONUS,
            ('down', 'sell'): TREND_ALIGNMENT_BONUS,
            ('up', 'sell'): -TREND_ALIGNMENT_BONUS * 0.5,
            ('down', 'buy'): -TREND_ALIGNMENT_BONUS * 0.5,
        }
        return alignment.get((trend, pattern_signal), 0.0)

    def _cnn_score(self, probs: np.ndarray) -> float:
        """
        Convert CNN class probabilities to a directional score.
        Assumes class layout: [sell, hold, buy] or similar.
        """
        if probs is None or len(probs) < 2:
            return 0.0
        if len(probs) >= 3:
            # Assume [sell_prob, hold_prob, buy_prob]
            sell_p, _, buy_p = probs[0], probs[1], probs[2]
        else:
            buy_p, sell_p = probs[0], probs[1]
        return float(buy_p - sell_p) * 0.5

    def _rule_alignment_score(self, rules: list[dict], pattern_signal: str) -> float:
        """Check active strategy rules alignment with detected signal."""
        if not rules:
            return 0.0
        aligned = sum(
            r.get('weight', 0.5)
            for r in rules
            if r.get('signal') == pattern_signal
        )
        total = sum(r.get('weight', 0.5) for r in rules)
        if total == 0:
            return 0.0
        return (aligned / total) * 0.15  # Max 0.15 boost from rules


# ─────────────────────────────────────────────
# Risk Calculator
# ─────────────────────────────────────────────

class RiskCalculator:
    """
    Calculates suggested entry, stop loss, take profit levels
    based on detected support/resistance zones and signal direction.
    """

    def calculate(
        self,
        signal: str,
        price_zones: list[dict],
        trend_angle: float,
        currency_pair: str = 'EURUSD',
    ) -> dict:
        """
        Calculate trade levels.

        Args:
            signal:      'buy' | 'sell' | 'hold'
            price_zones: list of {'y_pct': float, 'strength': float}
                         (y_pct = 0.0 at top of chart, 1.0 at bottom)
            trend_angle: detected chart trend angle in degrees
            currency_pair: for pip value calculation

        Returns:
            {
                'entry': float | None,
                'stop_loss': float | None,
                'take_profit': float | None,
                'risk_reward': float | None,
                'pip_distance_sl': float | None,
                'pip_distance_tp': float | None,
            }
        """
        if signal == 'hold' or not price_zones:
            return self._empty_levels()

        # Convert zone y_pct positions to normalized price levels
        # y_pct: 0.0 = top of chart (high price), 1.0 = bottom (low price)
        zone_prices = sorted(
            [1.0 - z['y_pct'] for z in price_zones],  # Invert: higher y_pct = lower price
            key=lambda z: z
        )

        if len(zone_prices) < 2:
            return self._empty_levels()

        # Simplified: use zone positions to estimate S/R levels
        # Real implementation would map y_pct to actual price using chart scale
        # Here we return normalized 0–1 levels as placeholders for the view layer
        support    = zone_prices[0]   # Lowest zone = support
        resistance = zone_prices[-1]  # Highest zone = resistance
        mid        = (support + resistance) / 2

        pip_scale = self._pip_scale(currency_pair)

        if signal == 'buy':
            entry       = round(mid, 5)
            stop_loss   = round(support - (mid - support) * 0.2, 5)
            take_profit = round(resistance + (resistance - mid) * 0.2, 5)
        else:  # sell
            entry       = round(mid, 5)
            stop_loss   = round(resistance + (resistance - mid) * 0.2, 5)
            take_profit = round(support - (mid - support) * 0.2, 5)

        risk   = abs(entry - stop_loss)
        reward = abs(entry - take_profit)
        rr     = round(reward / risk, 2) if risk > 0 else None

        return {
            'entry':           entry,
            'stop_loss':       stop_loss,
            'take_profit':     take_profit,
            'risk_reward':     rr,
            'pip_distance_sl': round(risk / pip_scale, 1) if pip_scale else None,
            'pip_distance_tp': round(reward / pip_scale, 1) if pip_scale else None,
        }

    def _pip_scale(self, pair: str) -> float:
        """Return pip size for common pairs."""
        yen_pairs = ['JPY', 'HUF', 'KRW']
        is_yen = any(p in pair.upper() for p in yen_pairs)
        return 0.01 if is_yen else 0.0001

    def _empty_levels(self) -> dict:
        return {
            'entry': None, 'stop_loss': None, 'take_profit': None,
            'risk_reward': None, 'pip_distance_sl': None, 'pip_distance_tp': None,
        }


# ─────────────────────────────────────────────
# Explanation Generator
# ─────────────────────────────────────────────

class ExplanationGenerator:
    """Generates human-readable AI reasoning for the prediction."""

    def generate(
        self,
        signal: str,
        confidence: float,
        patterns: list[dict],
        trend: str,
        timeframe: str,
        components: dict,
    ) -> str:
        """Returns a concise, trader-friendly explanation."""

        signal_text = {'buy': '📈 BUY', 'sell': '📉 SELL', 'hold': '⏸ HOLD / WAIT'}.get(signal, signal.upper())
        conf_pct = f"{confidence * 100:.0f}%"

        top_patterns = [p['name'] for p in patterns[:3]]
        pattern_str = ', '.join(top_patterns) if top_patterns else 'no clear pattern'

        trend_text = {
            'up': 'an uptrend', 'down': 'a downtrend', 'sideways': 'a sideways range'
        }.get(trend, 'an unclear trend')

        lines = [
            f"Signal: {signal_text} | Confidence: {conf_pct} | Timeframe: {timeframe}",
            "",
            f"The chart shows {trend_text}. "
            f"Detected patterns: {pattern_str}.",
        ]

        if signal != 'hold':
            dominant = 'bullish' if signal == 'buy' else 'bearish'
            lines.append(
                f"Pattern analysis is {dominant} ({components.get('pattern', 0):.0%} pattern score). "
                f"Trend alignment adds {'a bonus' if components.get('trend', 0) > 0 else 'some caution'}."
            )
            tf_weight = components.get('timeframe_weight', 0.75)
            lines.append(
                f"The {timeframe} timeframe carries {tf_weight:.0%} reliability weight in this analysis."
            )
        else:
            lines.append(
                "No strong directional consensus between patterns and trend. "
                "It is safer to wait for a clearer setup."
            )

        lines += [
            "",
            "⚠️ This is a probabilistic signal, not financial advice. "
            "Always apply your own risk management rules.",
        ]

        return '\n'.join(lines)


# ─────────────────────────────────────────────
# Main Predictor (Public API)
# ─────────────────────────────────────────────

class ForexPredictor:
    """
    Top-level prediction engine.
    Called by the Celery task `run_chart_analysis`.

    Usage:
        predictor = ForexPredictor()
        result = predictor.predict(
            image_path='/media/charts/eurusd_h1.png',
            currency_pair='EURUSD',
            timeframe='H1',
        )
    """

    MODEL_VERSION = 'v1.0'

    def __init__(self):
        ai_cfg = AI_CONFIG

        self.image_analyzer = ImageAnalyzer(
            model_path=str(ai_cfg.get('CHART_MODEL_PATH', ''))
        )
        self.pattern_detector = PatternDetector(
            cnn_model_path=str(ai_cfg.get('PATTERN_MODEL_PATH', ''))
        )
        self.signal_fusion    = SignalFusion()
        self.risk_calculator  = RiskCalculator()
        self.explainer        = ExplanationGenerator()

    def predict(
        self,
        image_path: str,
        currency_pair: str = 'EURUSD',
        timeframe: str = 'H1',
        candles: Optional[list] = None,
        active_rules: Optional[list] = None,
    ) -> dict:
        """
        Full prediction pipeline.

        Args:
            image_path:    Path to uploaded chart image
            currency_pair: e.g. 'EURUSD'
            timeframe:     e.g. 'H1'
            candles:       Optional list of Candle dicts for rule-based detection
            active_rules:  Optional list of active strategy rules from DB

        Returns full prediction dict consumed by AnalysisResult model.
        """
        logger.info(f"[Predictor] Starting prediction: {currency_pair} {timeframe}")

        # ── Step 1: Image analysis
        image_result = self.image_analyzer.analyze(image_path)
        visual       = image_result['visual']
        cnn_features = image_result['cnn_features']

        if visual['chart_quality'] < 0.25:
            logger.warning(f"[Predictor] Low chart quality: {visual['chart_quality']:.2f}")

        # ── Step 2: Pattern detection
        patterns = self.pattern_detector.detect(
            candles=candles,
            cnn_features=cnn_features,
        )
        pattern_signal, pattern_confidence = self.pattern_detector.aggregate_signal(patterns)

        # ── Step 3: Signal fusion
        fusion = self.signal_fusion.fuse(
            pattern_signal=pattern_signal,
            pattern_confidence=pattern_confidence,
            trend_direction=visual['trend_direction'],
            candle_ratio=visual['candle_ratio'],
            cnn_probs=cnn_features,
            timeframe=timeframe,
            active_rules=active_rules or [],
        )

        signal     = fusion['signal']
        confidence = fusion['confidence']

        # ── Step 4: Trend label
        trend_map = {
            'up': 'bullish', 'down': 'bearish',
            'sideways': 'sideways', 'ranging': 'ranging'
        }
        trend = trend_map.get(visual['trend_direction'], 'sideways')

        # ── Step 5: Support / Resistance from price zones
        zones = visual.get('price_zones', [])
        support_levels    = [z['y_pct'] for z in zones if z['y_pct'] > 0.5][:3]
        resistance_levels = [z['y_pct'] for z in zones if z['y_pct'] <= 0.5][:3]

        # ── Step 6: Risk / Reward calculation
        levels = self.risk_calculator.calculate(
            signal=signal,
            price_zones=zones,
            trend_angle=visual.get('trend_angle', 0),
            currency_pair=currency_pair,
        )

        # ── Step 7: Explanation
        explanation = self.explainer.generate(
            signal=signal,
            confidence=confidence,
            patterns=patterns,
            trend=visual['trend_direction'],
            timeframe=timeframe,
            components=fusion['components'],
        )

        result = {
            # Core outputs
            'signal':          signal,
            'trend':           trend,
            'confidence':      confidence,
            'patterns':        [p['slug'] for p in patterns],

            # Price levels
            'support_levels':    support_levels,
            'resistance_levels': resistance_levels,
            'entry':             levels['entry'],
            'stop_loss':         levels['stop_loss'],
            'take_profit':       levels['take_profit'],
            'risk_reward':       levels['risk_reward'],

            # Explanation
            'explanation': explanation,

            # Raw data for debugging
            'raw': {
                'visual_features':  visual,
                'pattern_details':  patterns,
                'fusion_components': fusion['components'],
                'levels_detail':    levels,
            },

            'model_version': self.MODEL_VERSION,
        }

        logger.info(
            f"[Predictor] Done: {currency_pair} {timeframe} → "
            f"signal={signal} confidence={confidence:.2%} "
            f"patterns={[p['slug'] for p in patterns[:3]]}"
        )

        return result