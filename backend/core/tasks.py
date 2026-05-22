"""
QuantNova AI — tasks.py
Celery async tasks: chart analysis, PDF extraction, model training, maintenance.
"""

import logging
import time
from celery import shared_task
from django.utils import timezone
from django.conf import settings

logger = logging.getLogger('apps')


# ─────────────────────────────────────────────
# CHART ANALYSIS TASK
# ─────────────────────────────────────────────

@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    name='apps.tasks.run_chart_analysis'
)
def run_chart_analysis(self, result_id: str):
    """
    Async task triggered when a user uploads a chart image.
    Runs the full AI analysis pipeline and saves results.

    Flow:
        ChartUpload saved → task queued → image_analyzer → predictor → save AnalysisResult
    """
    from .models import AnalysisResult
    from ai_engine.predictor import ForexPredictor

    try:
        result = AnalysisResult.objects.select_related('chart').get(id=result_id)
        result.status = 'processing'
        result.save(update_fields=['status'])

        logger.info(f"[ANALYSIS] Starting for result={result_id}, chart={result.chart.currency_pair}")
        start_time = time.time()

        image_path = result.chart.image.path
        predictor = ForexPredictor()
        prediction = predictor.predict(
            image_path=image_path,
            currency_pair=result.chart.currency_pair,
            timeframe=result.chart.timeframe,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Save all AI outputs to the result
        result.signal = prediction['signal']
        result.trend = prediction['trend']
        result.confidence = prediction['confidence']
        result.detected_patterns = prediction['patterns']
        result.support_levels = prediction['support_levels']
        result.resistance_levels = prediction['resistance_levels']
        result.suggested_entry = prediction.get('entry')
        result.suggested_sl = prediction.get('stop_loss')
        result.suggested_tp = prediction.get('take_profit')
        result.risk_reward_ratio = prediction.get('risk_reward')
        result.ai_explanation = prediction.get('explanation', '')
        result.raw_model_output = prediction.get('raw', {})
        result.processing_time_ms = elapsed_ms
        result.model_version = prediction.get('model_version', 'v1.0')
        result.status = 'completed'
        result.save()

        logger.info(
            f"[ANALYSIS] Completed result={result_id} | "
            f"signal={result.signal} | confidence={result.confidence:.2%} | "
            f"time={elapsed_ms}ms"
        )
        return {
            'status': 'completed',
            'result_id': result_id,
            'signal': result.signal,
            'confidence': result.confidence,
        }

    except AnalysisResult.DoesNotExist:
        logger.error(f"[ANALYSIS] AnalysisResult {result_id} not found.")
        return {'status': 'error', 'message': 'Result not found'}

    except Exception as exc:
        logger.error(f"[ANALYSIS] Failed for result={result_id}: {exc}", exc_info=True)
        try:
            result = AnalysisResult.objects.get(id=result_id)
            result.status = 'failed'
            result.save(update_fields=['status'])
        except Exception:
            pass
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────
# PDF EXTRACTION TASK
# ─────────────────────────────────────────────

@shared_task(
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    name='apps.tasks.run_pdf_extraction'
)
def run_pdf_extraction(self, document_id: str):
    """
    Async task for extracting trading rules from uploaded strategy PDFs.
    Uses NLP to parse rules, entry/exit conditions, and risk management notes.

    Flow:
        StrategyDocument saved → task queued → pdf_extractor → save ExtractedRules
    """
    from .models import StrategyDocument, ExtractedRule
    from ai_engine.pdf_extractor import PDFKnowledgeExtractor

    try:
        doc = StrategyDocument.objects.get(id=document_id)
        doc.status = 'processing'
        doc.save(update_fields=['status'])

        logger.info(f"[PDF] Starting extraction for document={document_id} '{doc.title}'")

        pdf_path = doc.pdf_file.path
        extractor = PDFKnowledgeExtractor()
        result = extractor.extract(pdf_path)

        doc.page_count = result['page_count']

        # Bulk-create extracted rules
        rules = [
            ExtractedRule(
                document=doc,
                rule_type=rule['type'],
                rule_text=rule['text'],
                confidence=rule['confidence'],
                page_number=rule.get('page'),
            )
            for rule in result['rules']
        ]
        ExtractedRule.objects.bulk_create(rules)

        doc.status = 'extracted'
        doc.save(update_fields=['status', 'page_count'])

        logger.info(
            f"[PDF] Extracted {len(rules)} rules from document={document_id} | "
            f"pages={result['page_count']}"
        )
        return {
            'status': 'extracted',
            'document_id': document_id,
            'rules_extracted': len(rules),
            'pages': result['page_count'],
        }

    except StrategyDocument.DoesNotExist:
        logger.error(f"[PDF] StrategyDocument {document_id} not found.")
        return {'status': 'error', 'message': 'Document not found'}

    except Exception as exc:
        logger.error(f"[PDF] Extraction failed for document={document_id}: {exc}", exc_info=True)
        try:
            doc = StrategyDocument.objects.get(id=document_id)
            doc.status = 'failed'
            doc.save(update_fields=['status'])
        except Exception:
            pass
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────
# MODEL TRAINING TASK
# ─────────────────────────────────────────────

@shared_task(
    bind=True,
    max_retries=1,
    name='apps.tasks.run_training_job',
    time_limit=3600,        # 1 hour hard limit
    soft_time_limit=3300,   # 55 min soft limit (graceful stop)
)
def run_training_job(self, job_id: str):
    """
    Async task for training/retraining an AI model.
    Long-running — may take minutes to hours depending on dataset size.

    Flow:
        TrainingJob created → task queued → trainer.py → save ModelVersion
    """
    from .models import TrainingJob, ModelVersion
    from ai_engine.trainer import ModelTrainer

    try:
        job = TrainingJob.objects.get(id=job_id)
        job.status = 'running'
        job.started_at = timezone.now()
        job.save(update_fields=['status', 'started_at'])

        logger.info(
            f"[TRAINING] Starting job={job_id} | "
            f"model={job.model_type} | epochs={job.epochs} | "
            f"batch={job.batch_size} | lr={job.learning_rate}"
        )

        trainer = ModelTrainer(
            model_type=job.model_type,
            epochs=job.epochs,
            batch_size=job.batch_size,
            learning_rate=job.learning_rate,
        )
        train_result = trainer.train()

        # Deactivate old versions of same model type
        ModelVersion.objects.filter(
            model_type=job.model_type, is_active=True
        ).update(is_active=False)

        # Save new model version
        import re
        last_version = (
            ModelVersion.objects
            .filter(model_type=job.model_type)
            .order_by('-created_at')
            .values_list('version_tag', flat=True)
            .first()
        )
        if last_version:
            nums = re.findall(r'\d+', last_version)
            new_tag = f"v{int(nums[0])}.{int(nums[1]) + 1}.0" if len(nums) >= 2 else 'v1.0.0'
        else:
            new_tag = 'v1.0.0'

        model_version = ModelVersion.objects.create(
            training_job=job,
            version_tag=new_tag,
            model_type=job.model_type,
            file_path=train_result['model_path'],
            accuracy=train_result['val_accuracy'],
            is_active=True,
            deployed_at=timezone.now(),
        )

        job.status = 'completed'
        job.completed_at = timezone.now()
        job.validation_accuracy = train_result['val_accuracy']
        job.training_loss = train_result['final_loss']
        job.training_samples = train_result['num_samples']
        job.model_file_path = train_result['model_path']
        job.save()

        logger.info(
            f"[TRAINING] Completed job={job_id} | "
            f"accuracy={train_result['val_accuracy']:.2%} | "
            f"version={new_tag} | samples={train_result['num_samples']}"
        )
        return {
            'status': 'completed',
            'job_id': job_id,
            'version': new_tag,
            'accuracy': train_result['val_accuracy'],
        }

    except TrainingJob.DoesNotExist:
        logger.error(f"[TRAINING] TrainingJob {job_id} not found.")
        return {'status': 'error', 'message': 'Job not found'}

    except Exception as exc:
        logger.error(f"[TRAINING] Job {job_id} failed: {exc}", exc_info=True)
        try:
            job = TrainingJob.objects.get(id=job_id)
            job.status = 'failed'
            job.error_log = str(exc)
            job.completed_at = timezone.now()
            job.save(update_fields=['status', 'error_log', 'completed_at'])
        except Exception:
            pass
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────
# SCHEDULED MAINTENANCE TASKS
# ─────────────────────────────────────────────

@shared_task(name='apps.tasks.auto_retrain_models')
def auto_retrain_models():
    """
    Scheduled task (weekly, Sunday 2AM) — auto-retrain models
    if enough new labeled data has accumulated since last training.

    Threshold: 100+ new chart analyses with confirmed outcomes.
    """
    from .models import TrainingJob, TradeJournal

    MIN_NEW_SAMPLES = 100
    model_types = ['chart_cnn', 'pattern_detector', 'signal_classifier']

    triggered = []
    for model_type in model_types:
        last_job = (
            TrainingJob.objects
            .filter(model_type=model_type, status='completed')
            .order_by('-completed_at')
            .first()
        )
        last_trained = last_job.completed_at if last_job else None

        # Count journal entries with confirmed AI outcomes since last training
        qs = TradeJournal.objects.filter(
            followed_ai_signal=True,
            ai_was_correct__isnull=False,
        )
        if last_trained:
            qs = qs.filter(closed_at__gte=last_trained)

        new_samples = qs.count()
        logger.info(f"[AUTO-TRAIN] {model_type}: {new_samples} new samples since last training")

        if new_samples >= MIN_NEW_SAMPLES:
            job = TrainingJob.objects.create(
                model_type=model_type,
                epochs=50,
                batch_size=32,
                learning_rate=0.0005,
                notes=f'Auto-triggered. New samples: {new_samples}',
            )
            run_training_job.delay(str(job.id))
            triggered.append(model_type)
            logger.info(f"[AUTO-TRAIN] Triggered retraining for {model_type} (job={job.id})")

    return {'triggered': triggered, 'skipped': [m for m in model_types if m not in triggered]}


@shared_task(name='apps.tasks.cleanup_old_pending_results')
def cleanup_old_pending_results():
    """
    Scheduled task (daily 3AM) — mark stale 'pending' or 'processing'
    results as 'failed' if they've been stuck for more than 30 minutes.
    This handles cases where a worker crashed mid-task.
    """
    from .models import AnalysisResult, StrategyDocument
    from datetime import timedelta

    cutoff = timezone.now() - timedelta(minutes=30)

    # Stuck analyses
    stuck_analyses = AnalysisResult.objects.filter(
        status__in=['pending', 'processing'],
        created_at__lt=cutoff
    )
    analysis_count = stuck_analyses.count()
    stuck_analyses.update(status='failed')

    # Stuck PDF extractions
    stuck_docs = StrategyDocument.objects.filter(
        status__in=['pending', 'processing'],
        uploaded_at__lt=cutoff
    )
    doc_count = stuck_docs.count()
    stuck_docs.update(status='failed')

    logger.info(
        f"[CLEANUP] Marked {analysis_count} stuck analyses and "
        f"{doc_count} stuck PDF extractions as failed."
    )
    return {
        'analyses_cleaned': analysis_count,
        'documents_cleaned': doc_count,
    }


@shared_task(name='apps.tasks.generate_daily_market_summary')
def generate_daily_market_summary():
    """
    Optional scheduled task — generates a brief AI summary of
    signal distribution across all pairs for the past 24 hours.
    Could be emailed to users or shown in the dashboard.
    """
    from .models import AnalysisResult
    from datetime import timedelta
    from django.db.models import Count, Avg

    since = timezone.now() - timedelta(hours=24)
    results = AnalysisResult.objects.filter(
        status='completed',
        created_at__gte=since
    )

    summary = {
        'period': '24h',
        'total_analyses': results.count(),
        'signals': {
            'buy': results.filter(signal='buy').count(),
            'sell': results.filter(signal='sell').count(),
            'hold': results.filter(signal='hold').count(),
        },
        'avg_confidence': results.aggregate(avg=Avg('confidence'))['avg'],
        'top_pairs': list(
            results.values('chart__currency_pair')
            .annotate(count=Count('id'), avg_conf=Avg('confidence'))
            .order_by('-count')[:5]
        ),
        'generated_at': timezone.now().isoformat(),
    }

    logger.info(f"[SUMMARY] Daily summary: {summary['total_analyses']} analyses processed")
    return summary