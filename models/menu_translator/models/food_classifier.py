"""
EfficientNet-based Food Classifier for dish recognition.

Uses timm (PyTorch Image Models) for model loading and inference.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None

try:
    from PIL import Image
    from torchvision import transforms
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    transforms = None

import numpy as np

from .data_models import DishPrediction, DishClassificationResult
from .metrics import metrics

logger = logging.getLogger(__name__)


# Food category mapping (subset of ImageNet classes related to food)
FOOD_CLASSES = {
    924: ("guacamole", "dip"),
    925: ("consomme", "soup"),
    926: ("hot pot", "stew"),
    927: ("trifle", "dessert"),
    928: ("ice cream", "dessert"),
    929: ("ice lolly", "dessert"),
    930: ("French loaf", "bread"),
    931: ("bagel", "bread"),
    932: ("pretzel", "bread"),
    933: ("cheeseburger", "sandwich"),
    934: ("hotdog", "sandwich"),
    935: ("mashed potato", "side"),
    936: ("head cabbage", "vegetable"),
    937: ("broccoli", "vegetable"),
    938: ("cauliflower", "vegetable"),
    939: ("zucchini", "vegetable"),
    940: ("spaghetti squash", "vegetable"),
    941: ("acorn squash", "vegetable"),
    942: ("butternut squash", "vegetable"),
    943: ("cucumber", "vegetable"),
    944: ("artichoke", "vegetable"),
    945: ("bell pepper", "vegetable"),
    946: ("cardoon", "vegetable"),
    947: ("mushroom", "vegetable"),
    948: ("Granny Smith", "fruit"),
    949: ("strawberry", "fruit"),
    950: ("orange", "fruit"),
    951: ("lemon", "fruit"),
    952: ("fig", "fruit"),
    953: ("pineapple", "fruit"),
    954: ("banana", "fruit"),
    955: ("jackfruit", "fruit"),
    956: ("custard apple", "fruit"),
    957: ("pomegranate", "fruit"),
    958: ("hay", "other"),
    959: ("carbonara", "pasta"),
    960: ("chocolate sauce", "sauce"),
    961: ("dough", "bread"),
    962: ("meat loaf", "meat"),
    963: ("pizza", "pizza"),
    964: ("potpie", "pie"),
    965: ("burrito", "mexican"),
    966: ("red wine", "beverage"),
    967: ("espresso", "beverage"),
    968: ("cup", "container"),
    969: ("eggnog", "beverage"),
}


@dataclass
class ClassifierConfig:
    """Configuration for food classifier."""
    model_name: str = "tf_efficientnet_b4"
    pretrained: bool = True
    num_classes: int = 1000  # ImageNet default
    top_k: int = 5
    device: str = "cpu"
    input_size: int = 380  # EfficientNet-B4 default
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)


class FoodClassifier:
    """
    EfficientNet-based food dish classifier.
    
    Uses pretrained EfficientNet-B4 from timm library for
    food image classification.
    
    Metrics tracked:
    - classifier_requests_total: Total classification requests
    - classifier_processing_seconds: Processing time histogram
    - classifier_top1_confidence: Top-1 prediction confidence
    - classifier_predictions: Prediction class counts
    - classifier_errors_total: Error counts
    - classifier_model_loaded: Model load status
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        """Initialize food classifier with configuration."""
        self.config = config or ClassifierConfig()
        self.model = None
        self.transform = None
        self._initialized = False
        self._class_names = None

        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        if not PIL_AVAILABLE:
            logger.warning("PIL/torchvision not available for transforms")
            return

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.config.input_size, self.config.input_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ),
        ])

    def load_model(self) -> bool:
        """
        Load the EfficientNet model.
        
        Returns:
            True if model loaded successfully
        """
        if not TIMM_AVAILABLE or not TORCH_AVAILABLE:
            logger.error("timm or torch not available")
            metrics.record_classifier_error("dependency_missing")
            return False

        try:
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
                num_classes=self.config.num_classes
            )
            self.model.eval()

            # Move to device
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()

            self._initialized = True
            metrics.set_model_loaded(True)
            logger.info(f"Loaded model: {self.config.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            metrics.record_classifier_error("model_load_error")
            return False

    def _load_imagenet_classes(self) -> list[str]:
        """Load ImageNet class names."""
        if self._class_names is not None:
            return self._class_names

        # Default to numbered classes if file not available
        self._class_names = [f"class_{i}" for i in range(1000)]

        # Try to load from URL (cached)
        try:
            import urllib.request
            url = (
                "https://raw.githubusercontent.com/pytorch/"
                "hub/master/imagenet_classes.txt"
            )
            with urllib.request.urlopen(url, timeout=5) as response:
                self._class_names = [
                    line.decode('utf-8').strip()
                    for line in response.readlines()
                ]
        except Exception:
            pass

        return self._class_names

    def classify(
        self,
        image: np.ndarray | Image.Image | bytes
    ) -> DishClassificationResult:
        """
        Classify a food image.
        
        Args:
            image: Input image as numpy array, PIL Image, or bytes
            
        Returns:
            DishClassificationResult with top-k predictions
        """
        start_time = time.time()
        metrics.increment_classifier_requests()

        predictions = []

        # Check if model is loaded
        if not self._initialized:
            if not self.load_model():
                return DishClassificationResult(
                    predictions=[],
                    processing_time_ms=0.0,
                    model_name=self.config.model_name
                )

        try:
            # Convert to PIL Image
            if isinstance(image, bytes):
                from io import BytesIO
                pil_image = Image.open(BytesIO(image)).convert('RGB')
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = image.convert('RGB')

            # Apply transforms
            if self.transform is None:
                self._setup_transforms()

            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0)

            # Move to device
            if self.config.device == "cuda" and torch.cuda.is_available():
                input_batch = input_batch.cuda()

            # Run inference
            with torch.inference_mode():
                output = self.model(input_batch)
                probabilities = F.softmax(output[0], dim=0)

            # Get top-k predictions
            top_k_prob, top_k_idx = torch.topk(
                probabilities, self.config.top_k
            )

            class_names = self._load_imagenet_classes()

            for rank, (prob, idx) in enumerate(
                zip(top_k_prob.cpu().numpy(), top_k_idx.cpu().numpy())
            ):
                idx_int = int(idx)
                class_name = class_names[idx_int]
                category = None

                # Check if it's a food class
                if idx_int in FOOD_CLASSES:
                    class_name, category = FOOD_CLASSES[idx_int]

                predictions.append(DishPrediction(
                    dish_class_id=str(idx_int),
                    dish_class_name=class_name,
                    confidence=float(prob),
                    category=category,
                    rank=rank
                ))

                metrics.record_prediction_class(class_name)

            # Record top-1 confidence
            if predictions:
                metrics.record_top1_confidence(predictions[0].confidence)

        except Exception as e:
            logger.error(f"Classification error: {e}")
            metrics.record_classifier_error("inference_error")

        processing_time = (time.time() - start_time) * 1000
        metrics.record_classifier_time(processing_time / 1000)

        return DishClassificationResult(
            predictions=predictions,
            processing_time_ms=processing_time,
            model_name=self.config.model_name
        )

    def calculate_accuracy(
        self,
        predictions: list[DishClassificationResult],
        ground_truths: list[str],
        k: int = 1
    ) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            predictions: List of classification results
            ground_truths: List of ground truth class IDs
            k: Top-k to consider
            
        Returns:
            Top-k accuracy (0.0 to 1.0)
        """
        if not predictions or not ground_truths:
            return 0.0

        correct = 0
        total = len(ground_truths)

        for pred, gt in zip(predictions, ground_truths):
            top_k_ids = [p.dish_class_id for p in pred.predictions[:k]]
            if gt in top_k_ids:
                correct += 1

        accuracy = correct / total

        if k == 1:
            metrics.record_top1_accuracy(accuracy)
        elif k == 5:
            metrics.record_top5_accuracy(accuracy)

        return accuracy

    def calculate_f1(
        self,
        predictions: list[str],
        ground_truths: list[str],
        average: str = "macro"
    ) -> float:
        """
        Calculate F1 score.
        
        Args:
            predictions: Predicted class IDs
            ground_truths: Ground truth class IDs
            average: 'macro', 'micro', or 'weighted'
            
        Returns:
            F1 score
        """
        from collections import Counter

        if not predictions or not ground_truths:
            return 0.0

        # Get all unique classes
        classes = set(predictions) | set(ground_truths)

        # Calculate per-class precision and recall
        f1_scores = []
        weights = []

        gt_counts = Counter(ground_truths)

        for cls in classes:
            tp = sum(
                1 for p, g in zip(predictions, ground_truths)
                if p == cls and g == cls
            )
            fp = sum(
                1 for p, g in zip(predictions, ground_truths)
                if p == cls and g != cls
            )
            fn = sum(
                1 for p, g in zip(predictions, ground_truths)
                if p != cls and g == cls
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            f1_scores.append(f1)
            weights.append(gt_counts.get(cls, 0))

        if average == "macro":
            result = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        elif average == "weighted":
            total_weight = sum(weights)
            if total_weight > 0:
                result = sum(
                    f * w for f, w in zip(f1_scores, weights)
                ) / total_weight
            else:
                result = 0.0
        else:  # micro
            tp_total = sum(
                1 for p, g in zip(predictions, ground_truths) if p == g
            )
            result = tp_total / len(predictions)

        metrics.record_f1_score(result)
        return result
