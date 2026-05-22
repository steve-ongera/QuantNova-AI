"""
QuantNova AI — models.py
Covers: users, analysis, journal, strategies, training
"""

import uuid
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone


# ─────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────

class User(AbstractUser):
    """Extended user with trader profile fields."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    preferred_pairs = models.JSONField(default=list, blank=True)  # e.g. ["EURUSD", "GBPJPY"]
    trading_style = models.CharField(
        max_length=50,
        choices=[
            ('scalper', 'Scalper'),
            ('day_trader', 'Day Trader'),
            ('swing', 'Swing Trader'),
            ('position', 'Position Trader'),
        ],
        default='day_trader'
    )
    risk_tolerance = models.CharField(
        max_length=20,
        choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')],
        default='medium'
    )
    timezone = models.CharField(max_length=50, default='Africa/Nairobi')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    class Meta:
        db_table = 'users'
        verbose_name = 'User'

    def __str__(self):
        return f"{self.email} ({self.trading_style})"


# ─────────────────────────────────────────────
# ANALYSIS — Chart uploads & AI predictions
# ─────────────────────────────────────────────

class ChartUpload(models.Model):
    """Stores a forex chart image uploaded by the user."""
    TIMEFRAME_CHOICES = [
        ('M1', '1 Minute'), ('M5', '5 Minutes'), ('M15', '15 Minutes'),
        ('M30', '30 Minutes'), ('H1', '1 Hour'), ('H4', '4 Hours'),
        ('D1', 'Daily'), ('W1', 'Weekly'), ('MN', 'Monthly'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chart_uploads')
    image = models.ImageField(upload_to='charts/%Y/%m/%d/')
    currency_pair = models.CharField(max_length=10, default='EURUSD')
    timeframe = models.CharField(max_length=5, choices=TIMEFRAME_CHOICES, default='H1')
    notes = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'chart_uploads'
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.currency_pair} {self.timeframe} — {self.user.email}"


class AnalysisResult(models.Model):
    """AI prediction result for a chart upload."""
    SIGNAL_CHOICES = [
        ('buy', 'BUY'),
        ('sell', 'SELL'),
        ('hold', 'HOLD / WAIT'),
    ]
    TREND_CHOICES = [
        ('bullish', 'Bullish'),
        ('bearish', 'Bearish'),
        ('sideways', 'Sideways'),
        ('ranging', 'Ranging'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chart = models.OneToOneField(ChartUpload, on_delete=models.CASCADE, related_name='result')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # AI outputs
    signal = models.CharField(max_length=10, choices=SIGNAL_CHOICES, null=True, blank=True)
    trend = models.CharField(max_length=20, choices=TREND_CHOICES, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)  # 0.0 – 1.0
    detected_patterns = models.JSONField(default=list, blank=True)
    # e.g. ["head_and_shoulders", "bullish_engulfing", "support_zone"]

    support_levels = models.JSONField(default=list, blank=True)   # [1.0820, 1.0795]
    resistance_levels = models.JSONField(default=list, blank=True)

    suggested_entry = models.FloatField(null=True, blank=True)
    suggested_sl = models.FloatField(null=True, blank=True)   # Stop Loss
    suggested_tp = models.FloatField(null=True, blank=True)   # Take Profit
    risk_reward_ratio = models.FloatField(null=True, blank=True)

    ai_explanation = models.TextField(blank=True)  # Human-readable AI reasoning
    raw_model_output = models.JSONField(default=dict, blank=True)  # Full model output

    processing_time_ms = models.IntegerField(null=True, blank=True)
    model_version = models.CharField(max_length=50, default='v1.0')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'analysis_results'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.chart.currency_pair} → {self.signal} ({self.confidence:.0%})"


class PatternLibrary(models.Model):
    """Reference library of candlestick/chart patterns the AI knows."""
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(unique=True)
    category = models.CharField(
        max_length=30,
        choices=[
            ('candlestick', 'Candlestick'),
            ('chart', 'Chart Pattern'),
            ('indicator', 'Indicator Signal'),
            ('smc', 'Smart Money Concept'),
        ]
    )
    description = models.TextField()
    bullish = models.BooleanField(default=True)  # True=bullish, False=bearish
    example_image = models.ImageField(upload_to='patterns/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'pattern_library'
        verbose_name_plural = 'Pattern Library'

    def __str__(self):
        return f"{self.name} ({'Bullish' if self.bullish else 'Bearish'})"


# ─────────────────────────────────────────────
# JOURNAL — Trade tracking & outcomes
# ─────────────────────────────────────────────

class TradeJournal(models.Model):
    """User's personal trade journal entry."""
    OUTCOME_CHOICES = [
        ('win', 'Win'),
        ('loss', 'Loss'),
        ('breakeven', 'Break Even'),
        ('open', 'Still Open'),
    ]
    DIRECTION_CHOICES = [('long', 'Long / Buy'), ('short', 'Short / Sell')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='journal_entries')
    analysis = models.ForeignKey(
        AnalysisResult, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='journal_entries'
    )

    currency_pair = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=5)
    direction = models.CharField(max_length=10, choices=DIRECTION_CHOICES)

    entry_price = models.FloatField()
    stop_loss = models.FloatField()
    take_profit = models.FloatField()
    lot_size = models.FloatField(default=0.01)
    risk_percent = models.FloatField(default=1.0)  # % of account risked

    outcome = models.CharField(max_length=15, choices=OUTCOME_CHOICES, default='open')
    pnl_pips = models.FloatField(null=True, blank=True)
    pnl_usd = models.FloatField(null=True, blank=True)

    followed_ai_signal = models.BooleanField(default=True)
    ai_was_correct = models.BooleanField(null=True, blank=True)  # set after outcome

    strategy_used = models.CharField(max_length=100, blank=True)
    emotional_state = models.CharField(
        max_length=30,
        choices=[
            ('calm', 'Calm'), ('excited', 'Excited'),
            ('fearful', 'Fearful'), ('greedy', 'Greedy'),
            ('disciplined', 'Disciplined'),
        ],
        default='calm'
    )
    notes = models.TextField(blank=True)
    screenshot = models.ImageField(upload_to='journal_screenshots/', null=True, blank=True)

    opened_at = models.DateTimeField(default=timezone.now)
    closed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'trade_journal'
        ordering = ['-opened_at']

    def __str__(self):
        return f"{self.currency_pair} {self.direction} — {self.outcome}"

    @property
    def risk_reward(self):
        if self.stop_loss and self.take_profit and self.entry_price:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            return round(reward / risk, 2) if risk else None
        return None


# ─────────────────────────────────────────────
# STRATEGIES — PDF knowledge base
# ─────────────────────────────────────────────

class StrategyDocument(models.Model):
    """Uploaded trading strategy PDF."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('extracted', 'Extracted'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='strategies')
    title = models.CharField(max_length=200)
    pdf_file = models.FileField(upload_to='strategies/%Y/%m/')
    description = models.TextField(blank=True)
    category = models.CharField(
        max_length=50,
        choices=[
            ('ict', 'ICT Concepts'),
            ('smc', 'Smart Money'),
            ('price_action', 'Price Action'),
            ('indicator', 'Indicator-Based'),
            ('risk', 'Risk Management'),
            ('psychology', 'Psychology'),
            ('custom', 'Custom Strategy'),
        ],
        default='custom'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    page_count = models.IntegerField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'strategy_documents'
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.title} [{self.category}]"


class ExtractedRule(models.Model):
    """A trading rule extracted from a strategy PDF by NLP."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(StrategyDocument, on_delete=models.CASCADE, related_name='rules')
    rule_type = models.CharField(
        max_length=40,
        choices=[
            ('entry', 'Entry Condition'),
            ('exit', 'Exit Condition'),
            ('risk', 'Risk Rule'),
            ('filter', 'Market Filter'),
            ('pattern', 'Pattern Signal'),
        ]
    )
    rule_text = models.TextField()
    confidence = models.FloatField(default=0.0)  # NLP extraction confidence
    page_number = models.IntegerField(null=True, blank=True)
    is_active = models.BooleanField(default=True)  # include in AI training
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'extracted_rules'
        ordering = ['rule_type', '-confidence']

    def __str__(self):
        return f"[{self.rule_type}] {self.rule_text[:80]}..."


# ─────────────────────────────────────────────
# TRAINING — AI model training management
# ─────────────────────────────────────────────

class TrainingJob(models.Model):
    """Tracks an AI model training run."""
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    MODEL_CHOICES = [
        ('chart_cnn', 'Chart CNN (Image Analysis)'),
        ('pattern_detector', 'Pattern Detector'),
        ('signal_classifier', 'Signal Classifier'),
        ('pdf_nlp', 'PDF NLP Extractor'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    triggered_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    model_type = models.CharField(max_length=30, choices=MODEL_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')

    epochs = models.IntegerField(default=50)
    batch_size = models.IntegerField(default=32)
    learning_rate = models.FloatField(default=0.001)

    training_samples = models.IntegerField(null=True, blank=True)
    validation_accuracy = models.FloatField(null=True, blank=True)
    training_loss = models.FloatField(null=True, blank=True)
    model_file_path = models.CharField(max_length=500, blank=True)
    notes = models.TextField(blank=True)
    error_log = models.TextField(blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'training_jobs'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.model_type} — {self.status} ({self.created_at.date()})"

    @property
    def duration_seconds(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).seconds
        return None


class ModelVersion(models.Model):
    """Deployed AI model versions."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    training_job = models.OneToOneField(TrainingJob, on_delete=models.CASCADE, related_name='version')
    version_tag = models.CharField(max_length=20, unique=True)  # e.g. "v1.3.2"
    model_type = models.CharField(max_length=30)
    file_path = models.CharField(max_length=500)
    accuracy = models.FloatField()
    is_active = models.BooleanField(default=False)  # only 1 active per type
    deployed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'model_versions'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.model_type} {self.version_tag} ({'active' if self.is_active else 'inactive'})"