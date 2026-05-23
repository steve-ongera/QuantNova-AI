"""
QuantNova AI — apps/signals.py
Django signals — post-save hooks for automation.
"""

import logging
from django.db.models.signals import post_save
from django.dispatch import receiver

logger = logging.getLogger('apps')


@receiver(post_save, sender='apps.TradeJournal')
def update_ai_accuracy_on_close(sender, instance, created, **kwargs):
    """
    When a journal trade is closed with an outcome and ai_was_correct is set,
    log AI accuracy feedback. Future: feed back into retraining pipeline.
    """
    if not created and instance.outcome != 'open' and instance.ai_was_correct is not None:
        result = 'correct ✓' if instance.ai_was_correct else 'incorrect ✗'
        logger.info(
            f"[Signal] Trade closed — AI signal was {result} | "
            f"pair={instance.currency_pair} outcome={instance.outcome}"
        )


@receiver(post_save, sender='apps.AnalysisResult')
def log_completed_analysis(sender, instance, created, **kwargs):
    """Log when an analysis result transitions to 'completed'."""
    if not created and instance.status == 'completed':
        logger.info(
            f"[Signal] Analysis completed — "
            f"pair={instance.chart.currency_pair} "
            f"signal={instance.signal} "
            f"confidence={instance.confidence:.2%}"
        )


@receiver(post_save, sender='apps.TrainingJob')
def log_training_completion(sender, instance, created, **kwargs):
    """Log when a training job completes."""
    if not created and instance.status == 'completed':
        logger.info(
            f"[Signal] Training job completed — "
            f"model={instance.model_type} "
            f"accuracy={instance.validation_accuracy:.2%}"
        )