"""
QuantNova AI — views.py
API views for all endpoints using DRF ViewSets + APIViews.
"""

from django.db.models import Avg, Count, Sum, Q
from django.utils import timezone
from rest_framework import generics, viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from .models import (
    User, ChartUpload, AnalysisResult, PatternLibrary,
    TradeJournal, StrategyDocument, ExtractedRule,
    TrainingJob, ModelVersion,
)
from .serializers import (
    RegisterSerializer, UserProfileSerializer,
    ChartUploadSerializer, AnalysisResultSerializer,
    AnalysisResultListSerializer, PatternLibrarySerializer,
    TradeJournalSerializer, TradeJournalStatsSerializer,
    StrategyDocumentSerializer, StrategyDocumentListSerializer,
    ExtractedRuleSerializer,
    TrainingJobSerializer, TriggerTrainingSerializer, ModelVersionSerializer,
)
from .tasks import run_chart_analysis, run_pdf_extraction, run_training_job


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────

class QuantNovaTokenSerializer(TokenObtainPairSerializer):
    """Custom JWT payload with extra user info."""
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['email'] = user.email
        token['trading_style'] = user.trading_style
        return token


class LoginView(TokenObtainPairView):
    serializer_class = QuantNovaTokenSerializer


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {'message': 'Account created. Welcome to QuantNova AI.', 'user_id': str(user.id)},
            status=status.HTTP_201_CREATED
        )


class ProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_object(self):
        return self.request.user


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

class ChartUploadViewSet(viewsets.ModelViewSet):
    """
    Upload forex chart images for AI analysis.
    POST /api/analysis/upload/ — triggers AI task via Celery.
    """
    serializer_class = ChartUploadSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        return ChartUpload.objects.filter(user=self.request.user).select_related('result')

    def perform_create(self, serializer):
        chart = serializer.save(user=self.request.user)
        # Create pending result and trigger async AI analysis
        result = AnalysisResult.objects.create(chart=chart, status='pending')
        run_chart_analysis.delay(str(result.id))

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(
            {
                'message': 'Chart uploaded. AI analysis started.',
                'chart_id': serializer.data['id'],
            },
            status=status.HTTP_202_ACCEPTED
        )


class AnalysisResultViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET /api/analysis/results/        — list all user's analysis results
    GET /api/analysis/results/{id}/   — get single result with full details
    """
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return AnalysisResult.objects.filter(
            chart__user=self.request.user
        ).select_related('chart')

    def get_serializer_class(self):
        if self.action == 'list':
            return AnalysisResultListSerializer
        return AnalysisResultSerializer

    @action(detail=False, methods=['get'], url_path='stats')
    def stats(self, request):
        """GET /api/analysis/results/stats/ — signal distribution stats."""
        qs = self.get_queryset().filter(status='completed')
        data = {
            'total': qs.count(),
            'buy_signals': qs.filter(signal='buy').count(),
            'sell_signals': qs.filter(signal='sell').count(),
            'hold_signals': qs.filter(signal='hold').count(),
            'avg_confidence': qs.aggregate(avg=Avg('confidence'))['avg'],
            'by_pair': list(
                qs.values('chart__currency_pair')
                .annotate(count=Count('id'), avg_conf=Avg('confidence'))
                .order_by('-count')[:10]
            ),
        }
        return Response(data)


class PatternLibraryView(generics.ListAPIView):
    """GET /api/analysis/patterns/ — reference pattern library."""
    queryset = PatternLibrary.objects.all().order_by('category', 'name')
    serializer_class = PatternLibrarySerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        category = self.request.query_params.get('category')
        bullish = self.request.query_params.get('bullish')
        if category:
            qs = qs.filter(category=category)
        if bullish is not None:
            qs = qs.filter(bullish=bullish.lower() == 'true')
        return qs


# ─────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────

class TradeJournalViewSet(viewsets.ModelViewSet):
    """
    CRUD for trade journal entries.
    GET  /api/journal/           — list entries
    POST /api/journal/           — create entry
    GET  /api/journal/{id}/      — get entry
    PUT  /api/journal/{id}/      — update entry (e.g. close trade)
    DEL  /api/journal/{id}/      — delete entry
    GET  /api/journal/stats/     — aggregated performance stats
    """
    serializer_class = TradeJournalSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_queryset(self):
        qs = TradeJournal.objects.filter(user=self.request.user).select_related('analysis')
        pair = self.request.query_params.get('pair')
        outcome = self.request.query_params.get('outcome')
        direction = self.request.query_params.get('direction')
        if pair:
            qs = qs.filter(currency_pair__iexact=pair)
        if outcome:
            qs = qs.filter(outcome=outcome)
        if direction:
            qs = qs.filter(direction=direction)
        return qs.order_by('-opened_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Aggregated trading performance statistics."""
        qs = TradeJournal.objects.filter(
            user=request.user, outcome__in=['win', 'loss', 'breakeven']
        )
        total = qs.count()
        wins = qs.filter(outcome='win').count()
        losses = qs.filter(outcome='loss').count()
        win_rate = (wins / total * 100) if total else 0
        total_pnl = qs.aggregate(total=Sum('pnl_usd'))['total'] or 0
        avg_rr = qs.aggregate(avg=Avg('risk_reward_ratio'))['avg'] or 0

        # Best pair by win rate
        pair_stats = (
            qs.values('currency_pair')
            .annotate(
                total=Count('id'),
                wins=Count('id', filter=Q(outcome='win'))
            )
        )
        best_pair = max(pair_stats, key=lambda x: x['wins'] / x['total'] if x['total'] else 0, default={})

        # AI accuracy
        ai_qs = qs.filter(followed_ai_signal=True, ai_was_correct__isnull=False)
        ai_correct = ai_qs.filter(ai_was_correct=True).count()
        ai_accuracy = (ai_correct / ai_qs.count() * 100) if ai_qs.count() else 0

        return Response({
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'total_pnl_usd': round(total_pnl, 2),
            'avg_rr': round(avg_rr, 2),
            'best_pair': best_pair.get('currency_pair', '—'),
            'ai_accuracy': round(ai_accuracy, 1),
        })

    @action(detail=True, methods=['post'], url_path='close')
    def close_trade(self, request, pk=None):
        """POST /api/journal/{id}/close/ — mark trade as closed with outcome."""
        trade = self.get_object()
        outcome = request.data.get('outcome')
        pnl_pips = request.data.get('pnl_pips')
        pnl_usd = request.data.get('pnl_usd')
        ai_was_correct = request.data.get('ai_was_correct')

        if outcome not in ['win', 'loss', 'breakeven']:
            return Response({'error': 'Invalid outcome.'}, status=status.HTTP_400_BAD_REQUEST)

        trade.outcome = outcome
        trade.pnl_pips = pnl_pips
        trade.pnl_usd = pnl_usd
        trade.ai_was_correct = ai_was_correct
        trade.closed_at = timezone.now()
        trade.save()
        return Response(TradeJournalSerializer(trade, context={'request': request}).data)


# ─────────────────────────────────────────────
# STRATEGIES
# ─────────────────────────────────────────────

class StrategyDocumentViewSet(viewsets.ModelViewSet):
    """
    Upload and manage PDF strategy documents.
    POST triggers NLP extraction via Celery.
    """
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_queryset(self):
        return StrategyDocument.objects.filter(user=self.request.user).prefetch_related('rules')

    def get_serializer_class(self):
        if self.action == 'list':
            return StrategyDocumentListSerializer
        return StrategyDocumentSerializer

    def perform_create(self, serializer):
        doc = serializer.save(user=self.request.user)
        run_pdf_extraction.delay(str(doc.id))

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(
            {'message': 'PDF uploaded. Knowledge extraction started.'},
            status=status.HTTP_202_ACCEPTED
        )

    @action(detail=True, methods=['get'])
    def rules(self, request, pk=None):
        """GET /api/strategies/{id}/rules/ — list extracted rules."""
        doc = self.get_object()
        rules = doc.rules.all().order_by('rule_type', '-confidence')
        return Response(ExtractedRuleSerializer(rules, many=True).data)

    @action(detail=True, methods=['post'], url_path='toggle-rule')
    def toggle_rule(self, request, pk=None):
        """POST /api/strategies/{id}/toggle-rule/ — activate/deactivate a rule."""
        rule_id = request.data.get('rule_id')
        try:
            rule = ExtractedRule.objects.get(id=rule_id, document__user=request.user)
            rule.is_active = not rule.is_active
            rule.save()
            return Response({'rule_id': rule_id, 'is_active': rule.is_active})
        except ExtractedRule.DoesNotExist:
            return Response({'error': 'Rule not found.'}, status=status.HTTP_404_NOT_FOUND)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

class TrainingJobViewSet(viewsets.ReadOnlyModelViewSet):
    """
    View training history.
    POST /api/training/trigger/ — start a new training job.
    GET  /api/training/status/  — current active job status.
    """
    serializer_class = TrainingJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return TrainingJob.objects.all().order_by('-created_at')

    @action(detail=False, methods=['post'])
    def trigger(self, request):
        """POST /api/training/trigger/ — queue a training job."""
        serializer = TriggerTrainingSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        job = TrainingJob.objects.create(
            triggered_by=request.user,
            model_type=serializer.validated_data['model_type'],
            epochs=serializer.validated_data['epochs'],
            batch_size=serializer.validated_data['batch_size'],
            learning_rate=serializer.validated_data['learning_rate'],
            notes=serializer.validated_data.get('notes', ''),
        )
        run_training_job.delay(str(job.id))
        return Response(
            {
                'message': 'Training job queued.',
                'job_id': str(job.id),
                'model_type': job.model_type,
            },
            status=status.HTTP_202_ACCEPTED
        )

    @action(detail=False, methods=['get'])
    def status(self, request):
        """GET /api/training/status/ — latest job per model type."""
        jobs = {}
        for model_type, _ in TrainingJob.MODEL_CHOICES:
            job = TrainingJob.objects.filter(model_type=model_type).order_by('-created_at').first()
            if job:
                jobs[model_type] = TrainingJobSerializer(job).data
        return Response(jobs)


class ModelVersionViewSet(viewsets.ReadOnlyModelViewSet):
    """GET /api/training/versions/ — list deployed model versions."""
    queryset = ModelVersion.objects.all().order_by('-created_at')
    serializer_class = ModelVersionSerializer
    permission_classes = [permissions.IsAuthenticated]

    @action(detail=False, methods=['get'], url_path='active')
    def active_versions(self, request):
        """GET /api/training/versions/active/ — only active models."""
        active = ModelVersion.objects.filter(is_active=True)
        return Response(ModelVersionSerializer(active, many=True).data)


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────

class DashboardView(generics.GenericAPIView):
    """
    GET /api/dashboard/
    Aggregated summary for the user's home dashboard.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user

        # Recent analyses
        recent_analyses = AnalysisResult.objects.filter(
            chart__user=user, status='completed'
        ).order_by('-created_at')[:5]

        # Journal quick stats
        closed_trades = TradeJournal.objects.filter(
            user=user, outcome__in=['win', 'loss']
        )
        total_trades = closed_trades.count()
        wins = closed_trades.filter(outcome='win').count()

        # Active model versions
        active_models = ModelVersion.objects.filter(is_active=True).values('model_type', 'version_tag', 'accuracy')

        return Response({
            'user': {
                'username': user.username,
                'trading_style': user.trading_style,
                'preferred_pairs': user.preferred_pairs,
            },
            'recent_analyses': AnalysisResultListSerializer(recent_analyses, many=True, context={'request': request}).data,
            'journal_summary': {
                'total_trades': total_trades,
                'win_rate': round((wins / total_trades * 100) if total_trades else 0, 1),
                'open_trades': TradeJournal.objects.filter(user=user, outcome='open').count(),
            },
            'active_models': list(active_models),
            'pending_analyses': AnalysisResult.objects.filter(
                chart__user=user, status__in=['pending', 'processing']
            ).count(),
        })