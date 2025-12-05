"""
Menu Translator - Training and Evaluation Script

This script demonstrates:
1. Training/fine-tuning the EfficientNet food classifier
2. Evaluating OCR with CER/WER metrics
3. Evaluating translation with BLEU scores
4. Running the full pipeline on sample data

Usage:
    python main.py --mode train --data-dir ./data
    python main.py --mode eval --data-dir ./data
    python main.py --mode demo
"""

import argparse
import logging
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np

# Import our models
from models import (
    OCRModel,
    OCRConfig,
    TranslationModel,
    FoodClassifier,
    ClassifierConfig,
    MenuParser,
    SupportedLanguage,
    metrics,
    HuggingFaceTranslationBackend,
    MarianMTBackend,
    MockTranslationBackend,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Sample Data for Demo/Testing
# ============================================================================

SAMPLE_MENU_ITEMS = [
    {"name": "비빔밥", "price": "12,000원", "lang": "ko"},
    {"name": "불고기", "price": "18,000원", "lang": "ko"},
    {"name": "Phở bò", "price": "85,000 VND", "lang": "vi"},
    {"name": "Bánh mì", "price": "35,000 VND", "lang": "vi"},
    {"name": "Grilled Salmon", "price": "$24.99", "lang": "en"},
    {"name": "Caesar Salad", "price": "$12.99", "lang": "en"},
]

SAMPLE_TRANSLATIONS = [
    {
        "source": "비빔밥",
        "source_lang": "ko",
        "reference_en": "bibimbap (mixed rice with vegetables)",
    },
    {
        "source": "phở",
        "source_lang": "vi",
        "reference_en": "pho (Vietnamese noodle soup)",
    },
]


# ============================================================================
# Training Functions
# ============================================================================

def train_food_classifier(
    data_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    output_dir: Optional[Path] = None
):
    """
    Fine-tune EfficientNet on food classification dataset.
    
    Expected data structure:
        data_dir/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    ...
            val/
                class1/
                    ...
    """
    logger.info("=" * 60)
    logger.info("Starting Food Classifier Training")
    logger.info("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import timm
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install torch torchvision timm")
        return None
    
    # Setup transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(380),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Load datasets
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists():
        logger.warning(f"Training directory not found: {train_dir}")
        logger.info("Creating sample directory structure...")
        _create_sample_data_structure(data_dir)
        return None
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    num_classes = len(train_dataset.classes)
    logger.info(f"Found {num_classes} classes: {train_dataset.classes}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = timm.create_model(
        "tf_efficientnet_b4",
        pretrained=True,
        num_classes=num_classes
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        scheduler.step()
        
        # Record metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        metrics.record_top1_accuracy(val_acc)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'classes': train_dataset.classes,
                }, output_dir / "best_model.pth")
                logger.info(f"Saved best model with acc: {val_acc:.4f}")
    
    logger.info(f"Training complete! Best accuracy: {best_acc:.4f}")
    return history


def _create_sample_data_structure(data_dir: Path):
    """Create sample directory structure for training."""
    classes = ["bibimbap", "pho", "pizza", "sushi", "tacos"]
    
    for split in ["train", "val"]:
        for cls in classes:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created sample structure at {data_dir}")
    logger.info("Add images to train/<class>/ and val/<class>/ directories")


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_ocr(data_dir: Path):
    """
    Evaluate OCR model with CER and WER metrics.
    
    Expected data structure:
        data_dir/
            ocr_eval/
                image1.jpg
                image1.txt  (ground truth)
                image2.jpg
                image2.txt
    """
    logger.info("=" * 60)
    logger.info("Evaluating OCR Model")
    logger.info("=" * 60)
    
    ocr_model = OCRModel(OCRConfig(use_gpu=False))
    
    eval_dir = data_dir / "ocr_eval"
    if not eval_dir.exists():
        logger.warning(f"OCR eval directory not found: {eval_dir}")
        # Run on synthetic data
        return _evaluate_ocr_synthetic(ocr_model)
    
    total_cer = 0.0
    total_wer = 0.0
    count = 0
    
    for img_path in eval_dir.glob("*.jpg"):
        gt_path = img_path.with_suffix(".txt")
        if not gt_path.exists():
            continue
        
        # Run OCR
        from PIL import Image
        image = Image.open(img_path)
        result = ocr_model.run_ocr(image)
        
        # Get ground truth
        ground_truth = gt_path.read_text().strip()
        predicted = " ".join([box.text for box in result.text_boxes])
        
        # Calculate metrics
        cer = ocr_model.calculate_cer(predicted, ground_truth)
        wer = ocr_model.calculate_wer(predicted, ground_truth)
        
        total_cer += cer
        total_wer += wer
        count += 1
        
        logger.info(f"{img_path.name}: CER={cer:.4f}, WER={wer:.4f}")
    
    if count > 0:
        avg_cer = total_cer / count
        avg_wer = total_wer / count
        logger.info(f"Average CER: {avg_cer:.4f}")
        logger.info(f"Average WER: {avg_wer:.4f}")
        return {"cer": avg_cer, "wer": avg_wer, "samples": count}
    
    return None


def _evaluate_ocr_synthetic(ocr_model: OCRModel):
    """Evaluate OCR on synthetic text pairs."""
    test_pairs = [
        ("hello world", "hello world"),  # Perfect match
        ("hello world", "helo world"),   # One error
        ("비빔밥 12000원", "비빔밥 12,000원"),  # Minor difference
        ("Phở bò đặc biệt", "Pho bo dac biet"),  # Diacritics
    ]
    
    results = []
    for predicted, ground_truth in test_pairs:
        cer = ocr_model.calculate_cer(predicted, ground_truth)
        wer = ocr_model.calculate_wer(predicted, ground_truth)
        results.append({"cer": cer, "wer": wer})
        logger.info(f"'{predicted}' vs '{ground_truth}': CER={cer:.4f}, WER={wer:.4f}")
    
    avg_cer = sum(r["cer"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)
    
    logger.info(f"Synthetic Avg CER: {avg_cer:.4f}")
    logger.info(f"Synthetic Avg WER: {avg_wer:.4f}")
    
    return {"cer": avg_cer, "wer": avg_wer, "samples": len(results)}


def evaluate_translation(data_dir: Path):
    """
    Evaluate translation model with BLEU scores.
    
    Expected data structure:
        data_dir/
            translation_eval.json
            [
                {"source": "text", "source_lang": "ko", "reference_en": "..."},
                ...
            ]
    """
    logger.info("=" * 60)
    logger.info("Evaluating Translation Model")
    logger.info("=" * 60)

    # Try to use real translation backend
    try:
        backend = HuggingFaceTranslationBackend(
            model_name="facebook/nllb-200-distilled-600M",
            device="auto"
        )
        logger.info("Using HuggingFace NLLB translation backend")
    except Exception as e:
        logger.warning(f"Could not load NLLB: {e}")
        logger.info("Falling back to mock translation backend")
        backend = MockTranslationBackend()

    translation_model = TranslationModel(backend=backend, use_glossary=True)
    
    eval_file = data_dir / "translation_eval.json"
    if eval_file.exists():
        eval_data = json.loads(eval_file.read_text())
    else:
        logger.info("Using sample translation data")
        eval_data = SAMPLE_TRANSLATIONS
    
    bleu_scores = []
    
    for item in eval_data:
        source = item["source"]
        source_lang = SupportedLanguage(item["source_lang"])
        reference = item.get("reference_en", "")
        
        # Translate
        result = translation_model.translate(
            source,
            source_lang,
            [SupportedLanguage.EN]
        )
        
        predicted = result.translated_text.en or ""
        
        # Calculate BLEU
        if reference:
            bleu = translation_model.calculate_bleu(predicted, reference)
            bleu_scores.append(bleu)
            logger.info(
                f"'{source}' -> '{predicted}' "
                f"(ref: '{reference}') BLEU={bleu:.4f}"
            )
    
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        logger.info(f"Average BLEU: {avg_bleu:.4f}")
        return {"bleu": avg_bleu, "samples": len(bleu_scores)}
    
    return None


def evaluate_classifier(data_dir: Path):
    """Evaluate food classifier on test set."""
    logger.info("=" * 60)
    logger.info("Evaluating Food Classifier")
    logger.info("=" * 60)
    
    classifier = FoodClassifier(ClassifierConfig(
        model_name="tf_efficientnet_b4",
        pretrained=True,
        device="cuda" if _cuda_available() else "cpu"
    ))
    
    test_dir = data_dir / "test"
    if not test_dir.exists():
        logger.warning(f"Test directory not found: {test_dir}")
        logger.info("Running classifier demo instead...")
        return _demo_classifier(classifier)
    
    try:
        from torchvision import datasets, transforms
        from PIL import Image
    except ImportError:
        logger.error("torchvision not available")
        return None
    
    transform = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    predictions = []
    ground_truths = []
    
    for idx, (image, label) in enumerate(test_dataset):
        # Convert tensor back to PIL for our classifier
        img_array = image.permute(1, 2, 0).numpy()
        img_array = (img_array * [0.229, 0.224, 0.225] + 
                     [0.485, 0.456, 0.406]) * 255
        img_array = img_array.astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        
        result = classifier.classify(pil_image)
        if result.predictions:
            predictions.append(result.predictions[0].dish_class_id)
        else:
            predictions.append("-1")
        ground_truths.append(str(label))
    
    # Calculate metrics
    top1_acc = classifier.calculate_accuracy(
        [classifier.classify(Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)))],
        ground_truths[:1],
        k=1
    )
    
    logger.info(f"Test samples: {len(ground_truths)}")
    logger.info(f"Top-1 Accuracy: {top1_acc:.4f}")
    
    return {"top1_acc": top1_acc, "samples": len(ground_truths)}


def _demo_classifier(classifier: FoodClassifier):
    """Run classifier on a dummy image."""
    logger.info("Creating dummy image for demo...")
    
    # Create a simple colored image
    dummy_image = np.random.randint(0, 255, (380, 380, 3), dtype=np.uint8)
    
    result = classifier.classify(dummy_image)
    
    logger.info(f"Model: {result.model_name}")
    logger.info(f"Processing time: {result.processing_time_ms:.2f}ms")
    logger.info("Top predictions:")
    for pred in result.predictions[:5]:
        logger.info(
            f"  {pred.rank+1}. {pred.dish_class_name} "
            f"({pred.confidence:.4f})"
        )
    
    return {"demo": True, "predictions": len(result.predictions)}


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================================
# Demo Pipeline
# ============================================================================

def run_demo(use_real_translation: bool = False):
    """Run a demo of the full pipeline."""
    logger.info("=" * 60)
    logger.info("Menu Translator Demo")
    logger.info("=" * 60)

    # Initialize models
    logger.info("Initializing models...")

    # Choose translation backend
    if use_real_translation:
        try:
            backend = HuggingFaceTranslationBackend(
                model_name="facebook/nllb-200-distilled-600M",
                device="auto"
            )
            logger.info("Using HuggingFace NLLB for real translation")
        except Exception as e:
            logger.warning(f"Could not load NLLB: {e}")
            backend = MockTranslationBackend()
    else:
        backend = MockTranslationBackend()
        logger.info("Using mock translation (use --real-translation for actual)")

    translation_model = TranslationModel(backend=backend, use_glossary=True)
    menu_parser = MenuParser()
    
    # Demo translations
    logger.info("\n--- Translation Demo ---")
    for item in SAMPLE_MENU_ITEMS:
        source_lang = SupportedLanguage(item["lang"])
        result = translation_model.translate(
            item["name"],
            source_lang,
            [SupportedLanguage.EN, SupportedLanguage.KO, SupportedLanguage.VI]
        )
        
        logger.info(f"\nOriginal ({item['lang']}): {item['name']}")
        logger.info(f"  EN: {result.translated_text.en}")
        logger.info(f"  KO: {result.translated_text.ko}")
        logger.info(f"  VI: {result.translated_text.vi}")
        if result.explanation.en:
            logger.info(f"  Explanation: {result.explanation.en}")
    
    # Demo menu parsing
    logger.info("\n--- Menu Parser Demo ---")
    from models.data_models import TextBox
    
    # Simulate OCR output
    mock_text_boxes = [
        TextBox(
            bbox=[10, 10, 200, 40],
            text="APPETIZERS",
            confidence=0.95,
            language=SupportedLanguage.EN
        ),
        TextBox(
            bbox=[10, 50, 150, 80],
            text="비빔밥",
            confidence=0.92,
            language=SupportedLanguage.KO
        ),
        TextBox(
            bbox=[160, 50, 250, 80],
            text="12,000원",
            confidence=0.88,
            language=SupportedLanguage.KO
        ),
        TextBox(
            bbox=[10, 100, 150, 130],
            text="Phở bò",
            confidence=0.90,
            language=SupportedLanguage.VI
        ),
        TextBox(
            bbox=[160, 100, 280, 130],
            text="85,000 VND",
            confidence=0.85,
            language=SupportedLanguage.VI
        ),
    ]
    
    menu_items = menu_parser.parse(mock_text_boxes)
    
    logger.info(f"\nExtracted {len(menu_items)} menu items:")
    for item in menu_items:
        logger.info(
            f"  - {item.raw_name} | "
            f"Price: {item.raw_price} | "
            f"Section: {item.section} | "
            f"Lang: {item.source_language.value}"
        )
    
    # Print metrics summary
    logger.info("\n--- Metrics Summary ---")
    all_metrics = metrics.get_metrics()
    for name, values in all_metrics.items():
        if values['count'] > 0:
            logger.info(f"{name}: count={values['count']}, avg={values['avg']:.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Menu Translator Training and Evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "demo", "all"],
        default="demo",
        help="Mode to run"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Output directory for models"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--real-translation",
        action="store_true",
        help="Use real HuggingFace translation (downloads ~600MB model)"
    )

    args = parser.parse_args()

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data dir: {args.data_dir}")

    if args.mode == "train":
        train_food_classifier(
            args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir
        )

    elif args.mode == "eval":
        evaluate_ocr(args.data_dir)
        evaluate_translation(args.data_dir)
        evaluate_classifier(args.data_dir)

    elif args.mode == "demo":
        run_demo(use_real_translation=args.real_translation)

    elif args.mode == "all":
        run_demo(use_real_translation=args.real_translation)
        evaluate_ocr(args.data_dir)
        evaluate_translation(args.data_dir)
        train_food_classifier(
            args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir
        )
    
    # Export metrics
    logger.info("\n--- Final Metrics ---")
    metrics_data = metrics.get_metrics()
    
    # Save metrics to file
    metrics_file = args.output_dir / "metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2, default=float)
    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
