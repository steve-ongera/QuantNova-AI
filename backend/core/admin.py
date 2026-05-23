"""
QuantNova AI — apps/admin.py
Django Admin configuration for all models.
Provides a clean interface for managing users, analyses, journal, strategies, and training.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone

from .models import (
    User, ChartUpload, AnalysisResult, PatternLibrary,
    TradeJournal, StrategyDocument, ExtractedRule,
    TrainingJob, ModelVersion,
)


# ─────────────────────────────────────────────
# Site Branding
# ─────────────────────────────────────────────

admin.site.site_header  = '⚡ QuantNova AI — Admin'
admin.site.site_title   = 'QuantNova Admin'
admin.site.index_title  = 'Platform Management'


# ─────────────────────────────────────────────
# USER
# ─────────────────────────────────────────────

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display   = ('email', 'username', 'trading_style', 'risk_tolerance', 'is_active', 'created_at')
    list_filter    = ('trading_style', 'risk_tolerance', 'is_active', 'is_staff')
    search_fields  = ('email', 'username')
    ordering       = ('-created_at',)
    readonly_fields = ('id', 'created_at', 'updated_at')

    fieldsets = (
        ('Account', {'fields': ('id', 'email', 'username', 'password')}),
        ('Trading Profile', {'fields': (
            'trading_style', 'risk_tolerance', 'preferred_pairs', 'timezone', 'avatar'
        )}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Timestamps', {'fields': ('created_at', 'updated_at', 'last_login')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'username', 'password1', 'password2', 'trading_style'),
        }),
    )


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

@admin.register(ChartUpload)
class ChartUploadAdmin(admin.ModelAdmin):
    list_display   = ('currency_pair', 'timeframe', 'user_email', 'uploaded_at', 'has_result')
    list_filter    = ('currency_pair', 'timeframe')
    search_fields  = ('user__email', 'currency_pair')
    ordering       = ('-uploaded_at',)
    readonly_fields = ('id', 'uploaded_at', 'chart_preview')

    def user_email(self, obj):
        return obj.user.email
    user_email.short_description = 'User'

    def has_result(self, obj):
        try:
            status = obj.result.status
            colors = {'completed': 'green', 'failed': 'red', 'processing': 'orange', 'pending': 'gray'}
            color  = colors.get(status, 'gray')
            return format_html('<span style="color:{}">●</span> {}', color, status)
        except AnalysisResult.DoesNotExist:
            return format_html('<span style="color:gray">— no result</span>')
    has_result.short_description = 'Status'

    def chart_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" style="max-width:300px; border-radius:4px;" />', obj.image.url)
        return '—'
    chart_preview.short_description = 'Preview'


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display   = ('pair_tf', 'signal_badge', 'confidence_display', 'trend', 'status', 'model_version', 'created_at')
    list_filter    = ('status', 'signal', 'trend', 'model_version')
    search_fields  = ('chart__currency_pair', 'chart__user__email')
    ordering       = ('-created_at',)
    readonly_fields = (
        'id', 'chart', 'signal', 'trend', 'confidence', 'detected_patterns',
        'support_levels', 'resistance_levels', 'suggested_entry', 'suggested_sl',
        'suggested_tp', 'risk_reward_ratio', 'ai_explanation', 'raw_model_output',
        'processing_time_ms', 'model_version', 'created_at', 'updated_at',
    )

    def pair_tf(self, obj):
        return f"{obj.chart.currency_pair} {obj.chart.timeframe}"
    pair_tf.short_description = 'Pair / TF'

    def signal_badge(self, obj):
        if not obj.signal:
            return '—'
        colors = {'buy': '#16a34a', 'sell': '#dc2626', 'hold': '#d97706'}
        color  = colors.get(obj.signal, '#6b7280')
        return format_html(
            '<span style="background:{};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{}</span>',
            color, obj.signal.upper()
        )
    signal_badge.short_description = 'Signal'

    def confidence_display(self, obj):
        if obj.confidence is None:
            return '—'
        pct   = obj.confidence * 100
        color = '#16a34a' if pct >= 70 else ('#d97706' if pct >= 55 else '#dc2626')
        return format_html(
            '<span style="color:{};font-weight:bold">{:.1f}%</span>',
            color, pct
        )
    confidence_display.short_description = 'Confidence'


@admin.register(PatternLibrary)
class PatternLibraryAdmin(admin.ModelAdmin):
    list_display  = ('name', 'category', 'bullish', 'slug')
    list_filter   = ('category', 'bullish')
    search_fields = ('name', 'slug')
    prepopulated_fields = {'slug': ('name',)}


# ─────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────

@admin.register(TradeJournal)
class TradeJournalAdmin(admin.ModelAdmin):
    list_display   = ('currency_pair', 'direction', 'outcome_badge', 'pnl_display', 'user_email', 'opened_at')
    list_filter    = ('outcome', 'direction', 'currency_pair', 'followed_ai_signal')
    search_fields  = ('user__email', 'currency_pair', 'strategy_used')
    ordering       = ('-opened_at',)
    readonly_fields = ('id', 'created_at', 'risk_reward')

    def user_email(self, obj):
        return obj.user.email
    user_email.short_description = 'User'

    def outcome_badge(self, obj):
        colors = {'win': '#16a34a', 'loss': '#dc2626', 'breakeven': '#d97706', 'open': '#6b7280'}
        color  = colors.get(obj.outcome, '#6b7280')
        return format_html(
            '<span style="color:{};font-weight:bold">{}</span>',
            color, obj.outcome.upper()
        )
    outcome_badge.short_description = 'Outcome'

    def pnl_display(self, obj):
        if obj.pnl_usd is None:
            return '—'
        color = '#16a34a' if obj.pnl_usd >= 0 else '#dc2626'
        return format_html('<span style="color:{}">${:.2f}</span>', color, obj.pnl_usd)
    pnl_display.short_description = 'P&L (USD)'


# ─────────────────────────────────────────────
# STRATEGIES
# ─────────────────────────────────────────────

class ExtractedRuleInline(admin.TabularInline):
    model   = ExtractedRule
    extra   = 0
    fields  = ('rule_type', 'confidence', 'page_number', 'is_active', 'rule_text')
    readonly_fields = ('rule_type', 'confidence', 'page_number', 'rule_text')
    can_delete = False
    ordering = ('rule_type', '-confidence')


@admin.register(StrategyDocument)
class StrategyDocumentAdmin(admin.ModelAdmin):
    list_display   = ('title', 'category', 'status_badge', 'rules_count', 'page_count', 'user_email', 'uploaded_at')
    list_filter    = ('status', 'category')
    search_fields  = ('title', 'user__email')
    ordering       = ('-uploaded_at',)
    readonly_fields = ('id', 'status', 'page_count', 'uploaded_at')
    inlines        = [ExtractedRuleInline]

    def user_email(self, obj):
        return obj.user.email
    user_email.short_description = 'User'

    def status_badge(self, obj):
        colors = {'extracted': 'green', 'failed': 'red', 'processing': 'orange', 'pending': 'gray'}
        color  = colors.get(obj.status, 'gray')
        return format_html('<span style="color:{}">● {}</span>', color, obj.status)
    status_badge.short_description = 'Status'

    def rules_count(self, obj):
        total  = obj.rules.count()
        active = obj.rules.filter(is_active=True).count()
        return f"{active} active / {total} total"
    rules_count.short_description = 'Rules'


@admin.register(ExtractedRule)
class ExtractedRuleAdmin(admin.ModelAdmin):
    list_display  = ('rule_type', 'confidence_display', 'page_number', 'is_active', 'document_title', 'short_text')
    list_filter   = ('rule_type', 'is_active')
    search_fields = ('rule_text', 'document__title')
    ordering      = ('rule_type', '-confidence')
    actions       = ['activate_rules', 'deactivate_rules']

    def document_title(self, obj):
        return obj.document.title
    document_title.short_description = 'Document'

    def short_text(self, obj):
        return obj.rule_text[:80] + '...' if len(obj.rule_text) > 80 else obj.rule_text
    short_text.short_description = 'Rule Preview'

    def confidence_display(self, obj):
        pct   = obj.confidence * 100
        color = '#16a34a' if pct >= 70 else ('#d97706' if pct >= 50 else '#dc2626')
        return format_html('<span style="color:{}">{:.0f}%</span>', color, pct)
    confidence_display.short_description = 'Confidence'

    @admin.action(description='Activate selected rules')
    def activate_rules(self, request, queryset):
        queryset.update(is_active=True)

    @admin.action(description='Deactivate selected rules')
    def deactivate_rules(self, request, queryset):
        queryset.update(is_active=False)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display   = ('model_type', 'status_badge', 'accuracy_display', 'training_samples', 'duration', 'triggered_by_email', 'created_at')
    list_filter    = ('status', 'model_type')
    search_fields  = ('model_type', 'triggered_by__email')
    ordering       = ('-created_at',)
    readonly_fields = (
        'id', 'status', 'training_samples', 'validation_accuracy',
        'training_loss', 'model_file_path', 'error_log',
        'started_at', 'completed_at', 'created_at',
    )

    def triggered_by_email(self, obj):
        return obj.triggered_by.email if obj.triggered_by else 'System'
    triggered_by_email.short_description = 'Triggered By'

    def status_badge(self, obj):
        colors = {'completed': 'green', 'failed': 'red', 'running': 'orange', 'queued': 'gray'}
        color  = colors.get(obj.status, 'gray')
        return format_html('<span style="color:{}">● {}</span>', color, obj.status)
    status_badge.short_description = 'Status'

    def accuracy_display(self, obj):
        if obj.validation_accuracy is None:
            return '—'
        pct   = obj.validation_accuracy * 100
        color = '#16a34a' if pct >= 70 else ('#d97706' if pct >= 55 else '#dc2626')
        return format_html('<span style="color:{};font-weight:bold">{:.1f}%</span>', color, pct)
    accuracy_display.short_description = 'Val. Accuracy'

    def duration(self, obj):
        secs = obj.duration_seconds
        if secs is None:
            return '—'
        if secs < 60:
            return f"{secs}s"
        return f"{secs // 60}m {secs % 60}s"
    duration.short_description = 'Duration'


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display  = ('version_tag', 'model_type', 'accuracy_display', 'is_active', 'deployed_at')
    list_filter   = ('model_type', 'is_active')
    ordering      = ('-created_at',)
    readonly_fields = ('id', 'training_job', 'created_at')
    actions       = ['set_as_active']

    def accuracy_display(self, obj):
        pct   = obj.accuracy * 100
        color = '#16a34a' if pct >= 70 else ('#d97706' if pct >= 55 else '#dc2626')
        return format_html('<span style="color:{};font-weight:bold">{:.1f}%</span>', color, pct)
    accuracy_display.short_description = 'Accuracy'

    @admin.action(description='Set selected version as active')
    def set_as_active(self, request, queryset):
        for version in queryset:
            # Deactivate others of same type
            ModelVersion.objects.filter(
                model_type=version.model_type, is_active=True
            ).update(is_active=False)
            version.is_active   = True
            version.deployed_at = timezone.now()
            version.save()
        self.message_user(request, f"Activated {queryset.count()} model version(s).")