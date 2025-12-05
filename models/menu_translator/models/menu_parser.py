"""
Menu Parser for grouping OCR text into menu items.

Handles menu structure detection, price extraction, and section identification.
"""

import re
import logging
from typing import Optional
from uuid import uuid4

from .data_models import TextBox, MenuItem, SupportedLanguage
from .metrics import metrics

logger = logging.getLogger(__name__)


# Price patterns for different currencies
PRICE_PATTERNS = [
    # Korean Won
    r'(\d{1,3}(?:,\d{3})*)\s*(?:₩|원|won)',
    r'₩\s*(\d{1,3}(?:,\d{3})*)',
    # Vietnamese Dong
    r'(\d{1,3}(?:[.,]\d{3})*)\s*(?:₫|VND|đ|dong)',
    r'(\d{1,3}(?:[.,]\d{3})*)\s*(?:k|K)\s*(?:₫|VND)?',
    # US Dollar
    r'\$\s*(\d+(?:\.\d{2})?)',
    r'(\d+(?:\.\d{2})?)\s*(?:USD|\$)',
    # Generic number that could be price
    r'^(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)$',
]

# Section header patterns
SECTION_PATTERNS = [
    # Common section names in English
    r'^(appetizers?|starters?|salads?|soups?|mains?|entrees?|'
    r'desserts?|drinks?|beverages?|sides?|specials?)$',
    # Korean section patterns
    r'^(전채|메인|디저트|음료|사이드|특선|추천).*$',
    # Vietnamese section patterns
    r'^(khai vị|món chính|tráng miệng|đồ uống|phụ).*$',
]


class MenuParser:
    """
    Parser for converting OCR results into structured menu items.
    
    Groups text boxes into menu items based on spatial proximity,
    detects prices, and identifies section headers.
    
    Metrics tracked:
    - parser_items_extracted: Number of menu items extracted
    - parser_sections_detected: Number of sections detected
    - parser_prices_detected: Number of prices detected
    - parser_processing_seconds: Processing time
    """

    def __init__(
        self,
        price_patterns: Optional[list[str]] = None,
        section_patterns: Optional[list[str]] = None,
        line_height_threshold: float = 50.0,
        column_threshold: float = 100.0
    ):
        """
        Initialize menu parser.
        
        Args:
            price_patterns: Custom regex patterns for prices
            section_patterns: Custom patterns for section headers
            line_height_threshold: Max Y distance for same line
            column_threshold: Max X distance for same column
        """
        self.price_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (price_patterns or PRICE_PATTERNS)
        ]
        self.section_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (section_patterns or SECTION_PATTERNS)
        ]
        self.line_height_threshold = line_height_threshold
        self.column_threshold = column_threshold

    def extract_price(self, text: str) -> Optional[str]:
        """
        Extract price from text.
        
        Returns:
            Extracted price string or None
        """
        for pattern in self.price_patterns:
            match = pattern.search(text)
            if match:
                metrics.increment_prices_detected()
                return match.group(0)
        return None

    def is_section_header(self, text: str) -> bool:
        """Check if text is a section header."""
        text_clean = text.strip()

        # Check patterns
        for pattern in self.section_patterns:
            if pattern.match(text_clean):
                return True

        # Heuristics: all caps, no numbers, short text
        if (text_clean.isupper() and
            len(text_clean) < 30 and
            not any(c.isdigit() for c in text_clean)):
            return True

        return False

    def _calculate_line_distance(
        self,
        box1: TextBox,
        box2: TextBox
    ) -> float:
        """Calculate vertical distance between two text boxes."""
        # Use center Y of each box
        y1_center = (box1.bbox[1] + box1.bbox[3]) / 2
        y2_center = (box2.bbox[1] + box2.bbox[3]) / 2
        return abs(y1_center - y2_center)

    def _calculate_column_distance(
        self,
        box1: TextBox,
        box2: TextBox
    ) -> float:
        """Calculate horizontal distance between two text boxes."""
        # Use left edge of each box
        return abs(box1.bbox[0] - box2.bbox[0])

    def _group_into_lines(
        self,
        text_boxes: list[TextBox]
    ) -> list[list[TextBox]]:
        """Group text boxes into lines based on Y position."""
        if not text_boxes:
            return []

        # Sort by Y position
        sorted_boxes = sorted(text_boxes, key=lambda b: b.bbox[1])

        lines = []
        current_line = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            if self._calculate_line_distance(
                current_line[0], box
            ) < self.line_height_threshold:
                current_line.append(box)
            else:
                # Sort current line by X position
                current_line.sort(key=lambda b: b.bbox[0])
                lines.append(current_line)
                current_line = [box]

        if current_line:
            current_line.sort(key=lambda b: b.bbox[0])
            lines.append(current_line)

        return lines

    def _merge_line_text(self, line: list[TextBox]) -> tuple[str, float]:
        """Merge text boxes in a line into single string."""
        texts = [box.text for box in line]
        confidences = [box.confidence for box in line]

        merged_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)

        return merged_text, avg_confidence

    def _get_line_bbox(self, line: list[TextBox]) -> list[float]:
        """Get bounding box for entire line."""
        x_min = min(box.bbox[0] for box in line)
        y_min = min(box.bbox[1] for box in line)
        x_max = max(box.bbox[2] for box in line)
        y_max = max(box.bbox[3] for box in line)
        return [x_min, y_min, x_max, y_max]

    def _get_line_language(self, line: list[TextBox]) -> SupportedLanguage:
        """Determine primary language of line."""
        lang_counts = {}
        for box in line:
            lang = box.language
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        if not lang_counts:
            return SupportedLanguage.OTHER

        return max(lang_counts, key=lang_counts.get)

    def parse(self, text_boxes: list[TextBox]) -> list[MenuItem]:
        """
        Parse OCR text boxes into structured menu items.
        
        Args:
            text_boxes: List of OCR detected text boxes
            
        Returns:
            List of parsed menu items
        """
        import time
        start_time = time.time()

        menu_items = []
        current_section = None

        # Group into lines
        lines = self._group_into_lines(text_boxes)

        for line in lines:
            merged_text, avg_confidence = self._merge_line_text(line)
            line_bbox = self._get_line_bbox(line)
            line_language = self._get_line_language(line)

            # Check if section header
            if self.is_section_header(merged_text):
                current_section = merged_text
                metrics.increment_sections_detected()
                continue

            # Extract price
            price = self.extract_price(merged_text)

            # Remove price from name if present
            name = merged_text
            if price:
                name = merged_text.replace(price, "").strip()
                # Clean up extra whitespace and punctuation
                name = re.sub(r'\s{2,}', ' ', name)
                name = name.strip(' .-–—')

            # Skip empty names
            if not name:
                continue

            # Create menu item
            menu_item = MenuItem(
                id=str(uuid4()),
                raw_name=name,
                raw_price=price,
                section=current_section,
                source_language=line_language,
                ocr_confidence=avg_confidence,
                bbox=line_bbox
            )
            menu_items.append(menu_item)

        # Record metrics
        metrics.record_items_extracted(len(menu_items))
        processing_time = (time.time() - start_time) * 1000
        metrics.record_parser_time(processing_time / 1000)

        return menu_items

    def calculate_grouping_accuracy(
        self,
        predicted_items: list[MenuItem],
        ground_truth_items: list[MenuItem]
    ) -> float:
        """
        Calculate menu item grouping accuracy.
        
        Compares predicted item names with ground truth.
        
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not ground_truth_items:
            return 0.0 if predicted_items else 1.0

        # Simple matching based on normalized names
        pred_names = {
            item.raw_name.lower().strip()
            for item in predicted_items
        }
        gt_names = {
            item.raw_name.lower().strip()
            for item in ground_truth_items
        }

        if not gt_names:
            return 0.0

        matches = len(pred_names & gt_names)
        accuracy = matches / len(gt_names)

        metrics.record_grouping_accuracy(accuracy)
        return accuracy
