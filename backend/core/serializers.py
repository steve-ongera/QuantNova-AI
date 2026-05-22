"""
QuantNova AI — serializers.py
DRF serializers for all models.
"""

from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import (
    ChartUpload, AnalysisResult, PatternLibrary,
    TradeJournal, StrategyDocument, ExtractedRule,
    TrainingJob, ModelVersion,
)

User = get_user_model()


# ─────────────────────────────────────────────
# AUTH / USER
# ─────────────────────────────────────────────

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = [
            'id', 'email', 'username', 'password', 'password_confirm',
            'trading_style', 'risk_tolerance', 'timezone',
        ]
        read_only_fields = ['id']

    def validate(self, data):
        if data['password'] != data.pop('password_confirm'):
            raise serializers.ValidationError({'password_confirm': 'Passwords do not match.'})
        return data

    def create(self, validated_data):
        user = User.objects.create_user(
            email=validated_data['email'],
            username=validated_data['username'],
            password=validated_data['password'],
            trading_style=validated_data.get('trading_style', 'day_trader'),
            risk_tolerance=validated_data.get('risk_tolerance', 'medium'),
            timezone=validated_data.get('timezone', 'Africa/Nairobi'),
        )
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'id', 'email', 'username', 'avatar',
            'preferred_pairs', 'trading_style', 'risk_tolerance',
            'timezone', 'created_at',
        ]
        read_only_fields = ['id', 'email', 'created_at']


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

class ChartUploadSerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())

    class Meta:
        model = ChartUpload
        fields = [
            'id', 'user', 'image', 'currency_pair',
            'timeframe', 'notes', 'uploaded_at',
        ]
        read_only_fields = ['id', 'uploaded_at']


class AnalysisResultSerializer(serializers.ModelSerializer):
    chart = ChartUploadSerializer(read_only=True)
    confidence_percent = serializers.SerializerMethodField()
    signal_display = serializers.CharField(source='get_signal_display', read_only=True)
    trend_display = serializers.CharField(source='get_trend_display', read_only=True)

    class Meta:
        model = AnalysisResult
        fields = [
            'id', 'chart', 'status',
            'signal', 'signal_display', 'trend', 'trend_display',
            'confidence', 'confidence_percent',
            'detected_patterns', 'support_levels', 'resistance_levels',
            'suggested_entry', 'suggested_sl', 'suggested_tp', 'risk_reward_ratio',
            'ai_explanation', 'processing_time_ms', 'model_version',
            'created_at', 'updated_at',
        ]
        read_only_fields = fields  # results are AI-generated, never user-editable

    def get_confidence_percent(self, obj):
        if obj.confidence is not None:
            return f"{obj.confidence * 100:.1f}%"
        return None


class AnalysisResultListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list views."""
    currency_pair = serializers.CharField(source='chart.currency_pair')
    timeframe = serializers.CharField(source='chart.timeframe')
    chart_image = serializers.ImageField(source='chart.image', read_only=True)
    confidence_percent = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisResult
        fields = [
            'id', 'currency_pair', 'timeframe', 'chart_image',
            'signal', 'trend', 'confidence', 'confidence_percent',
            'status', 'created_at',
        ]

    def get_confidence_percent(self, obj):
        if obj.confidence is not None:
            return round(obj.confidence * 100, 1)
        return None


class PatternLibrarySerializer(serializers.ModelSerializer):
    class Meta:
        model = PatternLibrary
        fields = '__all__'


# ─────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────

class TradeJournalSerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())
    risk_reward = serializers.ReadOnlyField()
    outcome_display = serializers.CharField(source='get_outcome_display', read_only=True)
    direction_display = serializers.CharField(source='get_direction_display', read_only=True)

    class Meta:
        model = TradeJournal
        fields = [
            'id', 'user', 'analysis',
            'currency_pair', 'timeframe', 'direction', 'direction_display',
            'entry_price', 'stop_loss', 'take_profit', 'lot_size', 'risk_percent',
            'outcome', 'outcome_display', 'pnl_pips', 'pnl_usd',
            'followed_ai_signal', 'ai_was_correct',
            'strategy_used', 'emotional_state', 'notes', 'screenshot',
            'risk_reward', 'opened_at', 'closed_at', 'created_at',
        ]
        read_only_fields = ['id', 'risk_reward', 'created_at']

    def validate(self, data):
        if data.get('outcome') != 'open' and not data.get('closed_at'):
            raise serializers.ValidationError(
                {'closed_at': 'Please provide close time for completed trades.'}
            )
        return data


class TradeJournalStatsSerializer(serializers.Serializer):
    """Aggregated stats for the journal dashboard."""
    total_trades = serializers.IntegerField()
    wins = serializers.IntegerField()
    losses = serializers.IntegerField()
    win_rate = serializers.FloatField()
    total_pnl_usd = serializers.FloatField()
    avg_rr = serializers.FloatField()
    best_pair = serializers.CharField()
    ai_accuracy = serializers.FloatField()


# ─────────────────────────────────────────────
# STRATEGIES
# ─────────────────────────────────────────────

class ExtractedRuleSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExtractedRule
        fields = [
            'id', 'rule_type', 'rule_text',
            'confidence', 'page_number', 'is_active', 'created_at',
        ]
        read_only_fields = ['id', 'confidence', 'created_at']


class StrategyDocumentSerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())
    rules = ExtractedRuleSerializer(many=True, read_only=True)
    rules_count = serializers.SerializerMethodField()
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    category_display = serializers.CharField(source='get_category_display', read_only=True)

    class Meta:
        model = StrategyDocument
        fields = [
            'id', 'user', 'title', 'pdf_file', 'description',
            'category', 'category_display', 'status', 'status_display',
            'page_count', 'rules', 'rules_count', 'uploaded_at',
        ]
        read_only_fields = ['id', 'status', 'page_count', 'uploaded_at']

    def get_rules_count(self, obj):
        return obj.rules.count()


class StrategyDocumentListSerializer(serializers.ModelSerializer):
    """Lightweight for list views."""
    rules_count = serializers.SerializerMethodField()
    category_display = serializers.CharField(source='get_category_display', read_only=True)

    class Meta:
        model = StrategyDocument
        fields = [
            'id', 'title', 'category', 'category_display',
            'status', 'rules_count', 'page_count', 'uploaded_at',
        ]

    def get_rules_count(self, obj):
        return obj.rules.filter(is_active=True).count()


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

class TrainingJobSerializer(serializers.ModelSerializer):
    triggered_by = serializers.StringRelatedField(read_only=True)
    duration_seconds = serializers.ReadOnlyField()
    model_type_display = serializers.CharField(source='get_model_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    validation_accuracy_percent = serializers.SerializerMethodField()

    class Meta:
        model = TrainingJob
        fields = [
            'id', 'triggered_by', 'model_type', 'model_type_display',
            'status', 'status_display',
            'epochs', 'batch_size', 'learning_rate',
            'training_samples', 'validation_accuracy', 'validation_accuracy_percent',
            'training_loss', 'model_file_path', 'notes', 'error_log',
            'duration_seconds', 'started_at', 'completed_at', 'created_at',
        ]
        read_only_fields = [
            'id', 'triggered_by', 'status', 'training_samples',
            'validation_accuracy', 'training_loss', 'model_file_path',
            'error_log', 'started_at', 'completed_at', 'created_at',
        ]

    def get_validation_accuracy_percent(self, obj):
        if obj.validation_accuracy is not None:
            return f"{obj.validation_accuracy * 100:.1f}%"
        return None


class TriggerTrainingSerializer(serializers.Serializer):
    """Input serializer for triggering a training job."""
    model_type = serializers.ChoiceField(choices=TrainingJob.MODEL_CHOICES)
    epochs = serializers.IntegerField(default=50, min_value=1, max_value=500)
    batch_size = serializers.IntegerField(default=32, min_value=8, max_value=256)
    learning_rate = serializers.FloatField(default=0.001, min_value=0.0001, max_value=0.1)
    notes = serializers.CharField(required=False, allow_blank=True)


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = [
            'id', 'version_tag', 'model_type', 'accuracy',
            'is_active', 'deployed_at', 'created_at',
        ]
        read_only_fields = fields