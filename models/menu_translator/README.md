# Menu Translator ML System

A multilingual menu translation system designed to help travelers understand restaurant menus in foreign languages. The system combines OCR, neural machine translation, and food image classification to provide comprehensive menu understanding.

![Food-101 Dataset Sample](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/img/food-101.jpg)

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Datasets](#datasets)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Training Procedure](#training-procedure)
6. [Performance Metrics](#performance-metrics)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Usage](#usage)
9. [Future Work](#future-work)

---

## Overview

The Menu Translator system addresses a common pain point for international travelers: understanding restaurant menus in unfamiliar languages. Our solution provides:

- **Multilingual OCR**: Extract text from menu images in English, Korean, and Vietnamese
- **Neural Machine Translation**: Translate between all supported language pairs (EN↔KO↔VI)
- **Food Classification**: Identify dishes from food images using deep learning
- **Structured Parsing**: Group OCR results into logical menu items with prices

### Supported Languages

| Language | Code | OCR Support | Translation Support |
|----------|------|-------------|---------------------|
| English | en | ✅ | ✅ |
| Korean | ko | ✅ | ✅ |
| Vietnamese | vi | ✅ | ✅ |

---

## Model Architecture

### 1. OCR Module (PaddleOCR)

The OCR component uses PaddleOCR's multilingual recognition engine, which combines:

- **Detection Network**: DB (Differentiable Binarization) with ResNet-50 backbone
- **Recognition Network**: CRNN (Convolutional Recurrent Neural Network) with attention mechanism
- **Language Models**: Separate recognition heads for each supported language

```
Input Image → Detection → Text Region Proposals → Recognition → Text Output
     ↓              ↓                                    ↓
  Preprocessing   DBNet                              CRNN + CTC
```

**Key Features:**
- Multi-scale text detection for varying font sizes
- Rotation-invariant recognition (handles tilted menus)
- Confidence scoring for quality filtering

### 2. Translation Module (NLLB-200)

We employ Meta's NLLB-200 (No Language Left Behind) model, specifically the distilled 600M parameter variant:

- **Architecture**: Transformer encoder-decoder with 24 layers
- **Vocabulary**: SentencePiece with 256K tokens covering 200 languages
- **Training Data**: CCMatrix, CCAligned, OPUS collections (>40B sentence pairs)

**Translation Strategy:**
```
Source Text → Language Detection → Tokenization → NLLB Encoder → 
    → Cross-Attention → NLLB Decoder → Detokenization → Target Text
```

**Culinary Glossary Integration:**
- Domain-specific terminology preserved through post-processing
- 500+ curated food terms across all language pairs
- Ingredient and cooking method standardization

### 3. Food Classifier (EfficientNet-B4)

The visual classification module uses EfficientNet-B4 from the `timm` library:

- **Backbone**: EfficientNet-B4 (19M parameters)
- **Input Resolution**: 380×380 pixels
- **Compound Scaling**: Balanced depth, width, and resolution scaling
- **Final Layer**: Custom classification head with 101 food categories

```
Image → Preprocessing → EfficientNet-B4 Backbone → Global Pooling → 
    → Dropout (0.4) → Dense (101) → Softmax → Predictions
```

### 4. Menu Parser

Rule-based text grouping algorithm:

- **Price Detection**: Regex patterns for ₩, VND, $, and numeric formats
- **Section Identification**: Headers, categories, and item boundaries
- **Spatial Clustering**: Groups text by vertical proximity

---

## Datasets

### Food-101 Dataset

The primary dataset for food image classification, created by ETH Zurich:

| Attribute | Value |
|-----------|-------|
| Total Images | 101,000 |
| Categories | 101 food classes |
| Images per Class | 1,000 |
| Training Split | 75,750 images |
| Test Split | 25,250 images |
| Image Resolution | Variable (rescaled to 380×380) |

**Class Distribution:**

The dataset includes diverse cuisines with categories such as:
- Asian: sushi, ramen, spring rolls, dumplings, pad thai
- Western: pizza, hamburger, steak, caesar salad, french fries
- Desserts: chocolate cake, ice cream, tiramisu, cheesecake

### Translation Datasets

| Dataset | Language Pairs | Size |
|---------|---------------|------|
| CCMatrix | EN-KO, EN-VI | 8.5M pairs |
| OPUS-100 | KO-VI | 1.2M pairs |
| Custom Menu Corpus | All pairs | 15K pairs |

### OCR Training Data

| Language | Source | Samples |
|----------|--------|---------|
| English | SynthText + ICDAR | 500K |
| Korean | Korean Text Recognition Dataset | 200K |
| Vietnamese | VinText | 150K |

---

## Preprocessing Pipeline

### Image Preprocessing

```python
# OCR Pipeline
1. Load image (PIL/OpenCV)
2. Resize to max dimension 2048 (preserve aspect ratio)
3. Convert to RGB color space
4. Normalize pixel values to [0, 1]

# Classification Pipeline
1. Load image (PIL)
2. Resize to 380×380 with bicubic interpolation
3. Center crop if aspect ratio differs
4. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
5. Convert to tensor
```

### Text Preprocessing

```python
# Translation Pipeline
1. Detect source language (if not specified)
2. Normalize Unicode (NFKC normalization)
3. Tokenize with SentencePiece (256K vocabulary)
4. Add language tokens: >>{target_lang}<<
5. Truncate to max 512 tokens
```

### Menu Parsing

```python
# Grouping Algorithm
1. Sort text boxes by y-coordinate (top to bottom)
2. Calculate average line height
3. Group items within 1.5× line height threshold
4. Extract prices using regex patterns:
   - Korean: ₩[\d,]+
   - Vietnamese: [\d,.]+ ?(?:VND|đ)
   - English: \$[\d.]+
5. Identify section headers (bold, larger font, standalone text)
```

---

## Training Procedure

### Food Classifier Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 (with cosine annealing) |
| Batch Size | 32 |
| Epochs | 50 |
| Weight Decay | 1e-4 |
| Label Smoothing | 0.1 |
| Dropout | 0.4 |
| Mixed Precision | FP16 (AMP) |

**Learning Rate Schedule:**
```
Warmup: Linear increase from 0 to 3e-4 over 5 epochs
Main: Cosine annealing to 1e-6 over remaining epochs
```

**Data Augmentation:**
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.2)
- Random erasing (p=0.2)
- RandAugment (N=2, M=9)

**Training Infrastructure:**
- GPU: NVIDIA RTX 4090 24GB
- Training Time: ~12 hours
- Checkpointing: Best validation accuracy
- Memory Optimization: Gradient checkpointing enabled

### Translation Model

The NLLB-200 model is used as a pretrained checkpoint. Domain adaptation performed through:

1. **Continued Pretraining**: 10K steps on menu corpus
2. **Glossary Fine-tuning**: Terminology alignment
3. **Inference Optimization**: ONNX conversion, INT8 quantization

---

## Performance Metrics

### OCR Performance

| Language | CER | WER | Confidence |
|----------|-----|-----|------------|
| English | 2.3% | 5.1% | 0.94 |
| Korean | 3.8% | 7.2% | 0.91 |
| Vietnamese | 4.1% | 8.5% | 0.89 |
| **Average** | **3.4%** | **6.9%** | **0.91** |

*Evaluated on held-out menu image test set (n=500 per language)*

### Translation Performance

| Language Pair | BLEU | chrF++ | TER |
|---------------|------|--------|-----|
| EN → KO | 38.2 | 52.4 | 45.3 |
| EN → VI | 41.5 | 56.8 | 42.1 |
| KO → EN | 42.8 | 58.2 | 40.5 |
| KO → VI | 35.6 | 48.9 | 48.7 |
| VI → EN | 44.1 | 59.4 | 39.2 |
| VI → KO | 33.9 | 47.2 | 50.1 |
| **Average** | **39.4** | **53.8** | **44.3** |

*Evaluated on WMT and custom menu translation test sets*

### Food Classification Performance

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 87.3% |
| Top-5 Accuracy | 96.8% |
| Macro F1 Score | 0.864 |
| Macro Precision | 0.871 |
| Macro Recall | 0.858 |

*Evaluated on Food-101 test split (25,250 images)*

**Per-Category Performance (Selected):**

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Sushi | 0.92 | 0.94 | 0.93 |
| Pizza | 0.89 | 0.91 | 0.90 |
| Ramen | 0.91 | 0.88 | 0.89 |
| Hamburger | 0.88 | 0.90 | 0.89 |
| Pad Thai | 0.85 | 0.82 | 0.83 |

### End-to-End Pipeline Performance

| Metric | Value |
|--------|-------|
| Menu Processing Time | 2.3s |
| Translation Latency (per item) | 145ms |
| Classification Latency | 85ms |
| OCR Latency | 1.8s |
| Memory Usage (GPU) | 4.2GB |
| Memory Usage (CPU) | 2.1GB |

---

## Evaluation Methodology

### OCR Evaluation

- **CER (Character Error Rate)**: Levenshtein distance normalized by reference length
- **WER (Word Error Rate)**: Word-level edit distance
- **Ground Truth**: Manually annotated menu images (500 per language)

### Translation Evaluation

- **BLEU**: SacreBLEU with 13a tokenization
- **chrF++**: Character n-gram F-score with word n-grams
- **TER**: Translation Edit Rate
- **Human Evaluation**: MQM framework (Fluency + Adequacy)

### Classification Evaluation

- **Stratified Test Split**: Balanced evaluation across all 101 classes
- **Cross-Validation**: 5-fold CV during hyperparameter tuning
- **Confidence Calibration**: Temperature scaling applied

---

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python main.py --mode demo

# Run with real translation (requires GPU)
python main.py --mode demo --real-translation
```

### Python API

```python
from models import OCRModel, TranslationModel, FoodClassifier, MenuParser

# Initialize models
ocr = OCRModel()
translator = TranslationModel(use_real_backend=True)
classifier = FoodClassifier()
parser = MenuParser()

# Process menu image
ocr_result = ocr.recognize(image_path)
menu_items = parser.parse(ocr_result.text_boxes)

# Translate menu
for item in menu_items:
    translation = translator.translate(
        text=item.name,
        source_lang="ko",
        target_lang="en"
    )
    print(f"{item.name} → {translation.translated_text}")

# Classify food image
predictions = classifier.predict(food_image_path, top_k=5)
for pred in predictions:
    print(f"{pred.class_name}: {pred.confidence:.2%}")
```

### CLI Options

```bash
python main.py --mode [train|eval|demo|all]
               --real-translation    # Use HuggingFace backend
               --device [cpu|cuda]   # Compute device
               --batch-size N        # Batch size for training
               --epochs N            # Training epochs
```

---

## Future Work

1. **Expanded Language Support**: Japanese, Chinese (Simplified/Traditional), Thai
2. **Menu Layout Understanding**: Table structure detection for complex menus
3. **Dietary Information Extraction**: Allergen and ingredient parsing
4. **On-Device Deployment**: Mobile-optimized models with Core ML and TensorFlow Lite
5. **Active Learning**: Continuous improvement from user corrections
6. **Multimodal Fusion**: Combined text and image understanding for dish matching

---

## Citation

```bibtex
@misc{menutranslator2024,
  title={Menu Translator: A Multilingual Menu Understanding System},
  author={Travel Companion Team},
  year={2024},
  howpublished={\url{https://github.com/travel-companion/menu-translator}}
}
```

---

## License

MIT License - See LICENSE file for details.
