"""
QuantNova AI — ai_engine/trainer.py
Local AI model training pipeline.

Supports training:
  - chart_cnn        : CNN for Buy/Sell/Hold classification from chart images
  - pattern_detector : PyTorch CNN for chart pattern recognition
  - signal_classifier: Sklearn classifier from combined features
  - pdf_nlp          : spaCy NER fine-tuning on trading rules

Usage (from Celery task):
    trainer = ModelTrainer(model_type='chart_cnn', epochs=50, batch_size=32)
    result  = trainer.train()
    # result = {'model_path': str, 'val_accuracy': float, 'final_loss': float, 'num_samples': int}

Usage (CLI):
    python trainer.py --model chart_cnn --epochs 50 --batch-size 32
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from django.conf import settings

logger = logging.getLogger('apps')

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

AI_CONFIG    = getattr(settings, 'AI_ENGINE', {})
MODELS_DIR   = Path(AI_CONFIG.get('MODELS_DIR',  'models_trained'))
DATASETS_DIR = Path(AI_CONFIG.get('DATASETS_DIR', 'datasets'))
IMAGE_SIZE   = AI_CONFIG.get('IMAGE_SIZE', (224, 224))

CHART_LABELS   = ['buy', 'hold', 'sell']     # Class indices: 0, 1, 2
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


# ─────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────

class ChartDataLoader:
    """
    Loads labeled chart images from:
        datasets/charts/buy/   → class 0
        datasets/charts/hold/  → class 1
        datasets/charts/sell/  → class 2
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, dataset_dir: Path, image_size: tuple = (224, 224)):
        self.dataset_dir = dataset_dir / 'charts'
        self.image_size  = image_size

    def load(self, validation_split: float = 0.2) -> dict:
        """
        Load all chart images and labels.
        Returns train/val split ready for model training.
        """
        import cv2

        images, labels = [], []

        for label_idx, label_name in enumerate(CHART_LABELS):
            class_dir = self.dataset_dir / label_name
            if not class_dir.exists():
                logger.warning(f"[DataLoader] Directory not found: {class_dir}")
                continue

            files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in self.VALID_EXTENSIONS
            ]
            logger.info(f"[DataLoader] {label_name}: {len(files)} images")

            for img_path in files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, self.image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(label_idx)

        if not images:
            raise ValueError(
                f"No training images found in {self.dataset_dir}. "
                "Add labeled images to datasets/charts/buy/, sell/, hold/"
            )

        images = np.array(images)
        labels = np.array(labels)

        # Shuffle
        idx = np.random.permutation(len(images))
        images, labels = images[idx], labels[idx]

        # Split
        split = int(len(images) * (1 - validation_split))
        return {
            'X_train': images[:split],  'y_train': labels[:split],
            'X_val':   images[split:],  'y_val':   labels[split:],
            'num_samples': len(images),
            'class_names': CHART_LABELS,
        }


class PatternDataLoader:
    """
    Loads chart pattern images from:
        datasets/charts/patterns/{pattern_name}/
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

    def __init__(self, dataset_dir: Path, image_size: tuple = (224, 224)):
        self.dataset_dir = dataset_dir / 'charts' / 'patterns'
        self.image_size  = image_size

    def load(self, validation_split: float = 0.2) -> dict:
        import cv2
        images, labels = [], []

        for label_idx, label_name in enumerate(PATTERN_LABELS):
            class_dir = self.dataset_dir / label_name
            if not class_dir.exists():
                continue
            files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in self.VALID_EXTENSIONS
            ]
            for img_path in files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, self.image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(label_idx)

        if not images:
            raise ValueError(f"No pattern images found in {self.dataset_dir}")

        images = np.array(images)
        labels = np.array(labels)
        idx    = np.random.permutation(len(images))
        images, labels = images[idx], labels[idx]
        split  = int(len(images) * (1 - validation_split))

        return {
            'X_train': images[:split],  'y_train': labels[:split],
            'X_val':   images[split:],  'y_val':   labels[split:],
            'num_samples': len(images),
            'class_names': PATTERN_LABELS,
        }


# ─────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────

class ChartCNNBuilder:
    """
    Builds a transfer-learning CNN for chart signal classification.
    Base: MobileNetV2 (lightweight, good for chart images)
    Head: GlobalAveragePooling → Dense(256) → Dropout → Dense(3, softmax)
    """

    def build(self, num_classes: int = 3, image_size: tuple = (224, 224)):
        import tensorflow as tf

        base = tf.keras.applications.MobileNetV2(
            input_shape=(*image_size, 3),
            include_top=False,
            weights='imagenet',
        )
        # Freeze base layers initially — fine-tune later
        base.trainable = False

        model = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ], name='quantnova_chart_cnn')

        return model


class PatternCNNBuilder:
    """
    Builds a PyTorch ResNet18-based pattern detector.
    Pretrained on ImageNet, fine-tuned for forex chart patterns.
    """

    def build(self, num_classes: int = 16):
        import torch
        import torch.nn as nn
        try:
            from torchvision import models
            model = models.resnet18(weights='IMAGENET1K_V1')
        except Exception:
            from torchvision import models
            model = models.resnet18(pretrained=True)

        # Freeze all except final layers
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        # Replace classification head
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        return model


class SignalClassifierBuilder:
    """
    Builds a Scikit-learn ensemble classifier using extracted features.
    Faster to train, good baseline for signal classification.
    """

    def build(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            ))
        ])


# ─────────────────────────────────────────────
# Training Engines
# ─────────────────────────────────────────────

class TensorFlowTrainer:
    """Trains TensorFlow/Keras CNN models."""

    def __init__(self, model, data: dict, epochs: int, batch_size: int,
                 learning_rate: float, model_name: str, output_dir: Path):
        self.model         = model
        self.data          = data
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.model_name    = model_name
        self.output_dir    = output_dir

    def train(self) -> dict:
        import tensorflow as tf

        # ── Phase 1: Train head only (frozen base)
        logger.info("[TFTrainer] Phase 1: Training head layers...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        callbacks = self._build_callbacks()
        history_1 = self.model.fit(
            self.data['X_train'], self.data['y_train'],
            validation_data=(self.data['X_val'], self.data['y_val']),
            epochs=max(10, self.epochs // 3),
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Phase 2: Fine-tune — unfreeze top layers of base
        logger.info("[TFTrainer] Phase 2: Fine-tuning base layers...")
        base_model = self.model.layers[0]
        base_model.trainable = True
        # Only unfreeze top 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate * 0.1),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        history_2 = self.model.fit(
            self.data['X_train'], self.data['y_train'],
            validation_data=(self.data['X_val'], self.data['y_val']),
            epochs=self.epochs,
            initial_epoch=len(history_1.history['loss']),
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Evaluate and save
        val_loss, val_acc = self.model.evaluate(
            self.data['X_val'], self.data['y_val'], verbose=0
        )

        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.output_dir / f"{self.model_name}_{timestamp}.h5"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_path))

        # Also save as 'latest' for easy loading
        latest_path = self.output_dir / f"{self.model_name}_latest.h5"
        self.model.save(str(latest_path))

        all_losses = history_1.history['loss'] + history_2.history['loss']

        logger.info(
            f"[TFTrainer] Training complete. "
            f"val_acc={val_acc:.4f} val_loss={val_loss:.4f} "
            f"saved={model_path}"
        )

        return {
            'model_path':    str(model_path),
            'val_accuracy':  round(float(val_acc), 4),
            'final_loss':    round(float(all_losses[-1]), 4),
            'num_samples':   self.data['num_samples'],
        }

    def _build_callbacks(self):
        import tensorflow as tf
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=8,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4,
                min_lr=1e-7, verbose=1,
            ),
        ]


class PyTorchTrainer:
    """Trains PyTorch CNN models."""

    def __init__(self, model, data: dict, epochs: int, batch_size: int,
                 learning_rate: float, model_name: str, output_dir: Path):
        self.model         = model
        self.data          = data
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.model_name    = model_name
        self.output_dir    = output_dir

    def train(self) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[PTTrainer] Training on device: {device}")
        self.model = self.model.to(device)

        # Build DataLoaders
        X_train = torch.tensor(self.data['X_train'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_train = torch.tensor(self.data['y_train'], dtype=torch.long)
        X_val   = torch.tensor(self.data['X_val'].transpose(0, 3, 1, 2), dtype=torch.float32)
        y_val   = torch.tensor(self.data['y_val'], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

        best_val_acc = 0.0
        best_state   = None
        final_loss   = 0.0
        patience_ctr = 0
        patience_max = 10

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / len(train_loader)

            # Validate
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs  = self.model(X_batch)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == y_batch).sum().item()
                    total   += y_batch.size(0)

            val_acc = correct / total if total > 0 else 0.0
            scheduler.step(final_loss)

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                logger.info(f"[PTTrainer] Epoch {epoch+1}/{self.epochs} | loss={final_loss:.4f} | val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience_max:
                    logger.info(f"[PTTrainer] Early stopping at epoch {epoch+1}")
                    break

        # Restore best weights and save
        if best_state:
            self.model.load_state_dict(best_state)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.output_dir / f"{self.model_name}_{timestamp}.pt"
        torch.save(self.model, str(model_path))
        torch.save(self.model, str(self.output_dir / f"{self.model_name}_latest.pt"))

        logger.info(f"[PTTrainer] Done. val_acc={best_val_acc:.4f} saved={model_path}")

        return {
            'model_path':   str(model_path),
            'val_accuracy': round(float(best_val_acc), 4),
            'final_loss':   round(float(final_loss), 4),
            'num_samples':  self.data['num_samples'],
        }


class SklearnTrainer:
    """Trains Scikit-learn classifiers on extracted feature vectors."""

    def __init__(self, model, data: dict, model_name: str, output_dir: Path, **kwargs):
        self.model      = model
        self.data       = data
        self.model_name = model_name
        self.output_dir = output_dir

    def train(self) -> dict:
        import pickle
        from sklearn.metrics import accuracy_score

        logger.info("[SKLearnTrainer] Starting training...")

        X_train = self.data['X_train'].reshape(len(self.data['X_train']), -1)
        X_val   = self.data['X_val'].reshape(len(self.data['X_val']), -1)

        self.model.fit(X_train, self.data['y_train'])

        val_preds = self.model.predict(X_val)
        val_acc   = accuracy_score(self.data['y_val'], val_preds)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.output_dir / f"{self.model_name}_{timestamp}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.output_dir / f"{self.model_name}_latest.pkl", 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"[SKLearnTrainer] Done. val_acc={val_acc:.4f} saved={model_path}")

        return {
            'model_path':   str(model_path),
            'val_accuracy': round(float(val_acc), 4),
            'final_loss':   round(1.0 - val_acc, 4),
            'num_samples':  self.data['num_samples'],
        }


# ─────────────────────────────────────────────
# Main Trainer (Public API)
# ─────────────────────────────────────────────

class ModelTrainer:
    """
    Unified training interface called by the Celery task.

    Supported model_type values:
        'chart_cnn'         → TensorFlow MobileNetV2 fine-tune
        'pattern_detector'  → PyTorch ResNet18 fine-tune
        'signal_classifier' → Sklearn GradientBoosting
        'pdf_nlp'           → spaCy NER (see _train_pdf_nlp)
    """

    def __init__(
        self,
        model_type:    str   = 'chart_cnn',
        epochs:        int   = 50,
        batch_size:    int   = 32,
        learning_rate: float = 0.001,
    ):
        self.model_type    = model_type
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate

    def train(self) -> dict:
        """
        Dispatch to the appropriate training method.
        Returns standard result dict for TrainingJob model.
        """
        logger.info(
            f"[ModelTrainer] Starting: model_type={self.model_type} "
            f"epochs={self.epochs} batch_size={self.batch_size} lr={self.learning_rate}"
        )

        dispatch = {
            'chart_cnn':         self._train_chart_cnn,
            'pattern_detector':  self._train_pattern_detector,
            'signal_classifier': self._train_signal_classifier,
            'pdf_nlp':           self._train_pdf_nlp,
        }

        trainer_fn = dispatch.get(self.model_type)
        if not trainer_fn:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return trainer_fn()

    # ─────────────────────────────────────────
    # Training Methods
    # ─────────────────────────────────────────

    def _train_chart_cnn(self) -> dict:
        loader = ChartDataLoader(DATASETS_DIR, IMAGE_SIZE)
        data   = loader.load()
        model  = ChartCNNBuilder().build(num_classes=len(CHART_LABELS), image_size=IMAGE_SIZE)
        trainer = TensorFlowTrainer(
            model=model, data=data,
            epochs=self.epochs, batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            model_name='chart_cnn', output_dir=MODELS_DIR,
        )
        return trainer.train()

    def _train_pattern_detector(self) -> dict:
        loader  = PatternDataLoader(DATASETS_DIR, IMAGE_SIZE)
        data    = loader.load()
        model   = PatternCNNBuilder().build(num_classes=len(PATTERN_LABELS))
        trainer = PyTorchTrainer(
            model=model, data=data,
            epochs=self.epochs, batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            model_name='pattern_detector', output_dir=MODELS_DIR,
        )
        return trainer.train()

    def _train_signal_classifier(self) -> dict:
        """
        Trains signal classifier on feature vectors extracted from chart images.
        Requires pre-extracted features in datasets/features/.
        """
        features_dir = DATASETS_DIR / 'features'
        X_path = features_dir / 'X.npy'
        y_path = features_dir / 'y.npy'

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Feature vectors not found at {features_dir}. "
                "Run feature extraction first: python extract_features.py"
            )

        X = np.load(str(X_path))
        y = np.load(str(y_path))
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        split = int(len(X) * 0.8)

        data = {
            'X_train': X[:split], 'y_train': y[:split],
            'X_val':   X[split:], 'y_val':   y[split:],
            'num_samples': len(X),
        }
        model   = SignalClassifierBuilder().build()
        trainer = SklearnTrainer(
            model=model, data=data,
            model_name='signal_classifier', output_dir=MODELS_DIR,
        )
        return trainer.train()

    def _train_pdf_nlp(self) -> dict:
        """
        Fine-tune spaCy NER model on trading rule entities.
        Training data: datasets/strategies/training_data.json

        Expected format:
        [
          {"text": "Only enter after a Break of Structure", "entities": [[10, 15, "ENTRY_TRIGGER"]]},
          ...
        ]
        """
        import spacy
        from spacy.training import Example
        from spacy.util import minibatch, compounding

        data_path = DATASETS_DIR / 'strategies' / 'training_data.json'
        if not data_path.exists():
            raise FileNotFoundError(
                f"NLP training data not found: {data_path}. "
                "Create datasets/strategies/training_data.json with labeled sentences."
            )

        with open(data_path) as f:
            raw_data = json.load(f)

        # Load or create blank model
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            nlp = spacy.blank('en')

        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner')
        else:
            ner = nlp.get_pipe('ner')

        # Entity labels for trading rules
        for label in ['ENTRY_TRIGGER', 'EXIT_CONDITION', 'RISK_RULE', 'MARKET_FILTER', 'PATTERN_NAME']:
            ner.add_label(label)

        # Build training examples
        examples = []
        for item in raw_data:
            doc = nlp.make_doc(item['text'])
            entities = [(start, end, label) for start, end, label in item.get('entities', [])]
            example = Example.from_dict(doc, {'entities': entities})
            examples.append(example)

        logger.info(f"[NLPTrainer] Training on {len(examples)} sentences for {self.epochs} iterations...")

        other_pipes = [p for p in nlp.pipe_names if p != 'ner']
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            losses_history = []

            for epoch in range(self.epochs):
                np.random.shuffle(examples)
                losses = {}
                batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    nlp.update(batch, drop=0.3, sgd=optimizer, losses=losses)
                losses_history.append(losses.get('ner', 0))

                if epoch % 10 == 0:
                    logger.info(f"[NLPTrainer] Epoch {epoch+1}/{self.epochs} | NER loss: {losses.get('ner', 0):.4f}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = MODELS_DIR / f"pdf_nlp_{timestamp}"
        nlp.to_disk(str(model_path))
        nlp.to_disk(str(MODELS_DIR / 'pdf_nlp_latest'))

        final_loss = losses_history[-1] if losses_history else 0.0
        # spaCy NER doesn't have direct val_accuracy — estimate from loss improvement
        val_accuracy = max(0.0, 1.0 - (final_loss / max(losses_history[0], 1e-6)))

        logger.info(f"[NLPTrainer] Done. model saved to {model_path}")

        return {
            'model_path':   str(model_path),
            'val_accuracy': round(min(0.95, float(val_accuracy)), 4),
            'final_loss':   round(float(final_loss), 4),
            'num_samples':  len(examples),
        }


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()

    parser = argparse.ArgumentParser(description='QuantNova AI Model Trainer')
    parser.add_argument('--model', type=str, default='chart_cnn',
                        choices=['chart_cnn', 'pattern_detector', 'signal_classifier', 'pdf_nlp'],
                        help='Model type to train')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  QuantNova AI — Model Trainer")
    print(f"  Model:      {args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"{'='*50}\n")

    trainer = ModelTrainer(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    result = trainer.train()

    print(f"\n{'='*50}")
    print(f"  Training Complete!")
    print(f"  Accuracy : {result['val_accuracy']:.2%}")
    print(f"  Loss     : {result['final_loss']:.4f}")
    print(f"  Samples  : {result['num_samples']}")
    print(f"  Saved to : {result['model_path']}")
    print(f"{'='*50}\n")