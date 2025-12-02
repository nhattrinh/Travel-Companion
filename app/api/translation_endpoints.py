"""Translation endpoints for live frame and save actions."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import time
import os

from app.services.ocr_service import OCRService
from app.services.lang_detect import detect_language
from app.services.translation_service import MockTranslationModel
from app.core.db import db_session
from app.services.translation_history_service import TranslationHistoryService
from app.services.image_preprocess import enhance_contrast
from app.schemas.translation import LiveFrameResponse, LiveFrameSegment
from app.core.metrics_translation import record_translation_latency
from app.core.deprecation import DeprecationMapper
import logging

logger = logging.getLogger(__name__)
mapper = DeprecationMapper({"segments_legacy": "segments"})

router = APIRouter(prefix="/translation", tags=["translation"])

ocr_service = OCRService()
translation_model = MockTranslationModel()
history_service = TranslationHistoryService(db_session)


class TextTranslationRequest(BaseModel):
    """Request model for text translation"""
    text: str
    target_language: str = "en"
    source_language: Optional[str] = None


class TextTranslationResponse(BaseModel):
    """Response model for text translation"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float


@router.post("/text")
async def translate_text(request: TextTranslationRequest):
    """
    Translate text from one language to another.
    Auto-detects source language if not provided.
    """
    try:
        start = time.perf_counter()
        
        # Detect source language if not provided
        source_lang = request.source_language
        if not source_lang:
            source_lang = detect_language(request.text)
        
        # Translate the text
        result = await translation_model.translate(
            request.text,
            target_language=request.target_language,
            source_language=source_lang,
        )
        
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(f"Text translation completed in {latency_ms:.2f}ms")
        
        response_data = TextTranslationResponse(
            original_text=request.text,
            translated_text=result["translated_text"],
            source_language=source_lang,
            target_language=request.target_language,
            confidence=result.get("confidence", 0.95),
        )
        
        return {"status": "ok", "data": response_data.dict(), "error": None}
        
    except Exception as e:
        logger.error(f"Text translation failed: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "data": None,
                "error": f"TRANSLATION_FAILED: {str(e)}",
            },
        )


@router.post("/live-frame")
async def translate_live_frame(
    image: UploadFile = File(...),
    target_language: str = "en",
):
    try:
        with record_translation_latency():
            raw = await image.read()
        start = time.perf_counter()
        # basic preprocess
        processed = enhance_contrast(raw)
        
        # Create OCR service with Gemini Vision for OCR + translation + images
        from app.services.ocr_service import OCRService, GeminiVisionOCRModel
        if os.getenv("GOOGLE_API_KEY"):
            ocr = OCRService(GeminiVisionOCRModel(target_language=target_language))
        else:
            ocr = ocr_service
        
        ocr_results = await ocr.extract_text(processed)
        source_lang = detect_language(" ".join(r.text for r in ocr_results))
        segments: list[LiveFrameSegment] = []
        
        for r in ocr_results:
            # Use pre-translated text from Gemini Vision if available
            if r.translated_text and r.translated_text != r.text:
                translated_text = r.translated_text
            else:
                # Fallback to translation model
                tr = await translation_model.translate(
                    r.text,
                    target_language=target_language,
                    source_language=source_lang,
                )
                translated_text = tr["translated_text"]
            
            # Use image_url from Gemini response
            photo_url = r.image_url if r.item_type != "price" else None
            
            segments.append(
                LiveFrameSegment(
                    text=r.text,
                    translated=translated_text,
                    x1=r.bbox[0],
                    y1=r.bbox[1],
                    x2=r.bbox[2],
                    y2=r.bbox[3],
                    confidence=r.confidence,
                    item_type=r.item_type or "food",
                    price=r.price,
                    photo_url=photo_url,
                )
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        result = LiveFrameResponse(
            segments=segments,
            source_language=source_lang,
            target_language=target_language,
            latency_ms=latency_ms,
        )
        data_payload = result.dict()
        # Add deprecated field names via mapper
        data_payload = mapper.transform_outbound(data_payload)
        deprecated_used = mapper.audit(data_payload)
        if deprecated_used:
            logger.info(
                "deprecated_fields_outbound",
                extra={"fields": deprecated_used},
            )
        return {"status": "ok", "data": data_payload, "error": None}
    except Exception as e:
        logger.exception("OCR/translation failed")
        # Envelope-style error for OCR/processing failures
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "data": None,
                "error": f"OCR_PROCESSING_FAILED: {str(e)}",
            },
        )

 
@router.post("/save")
async def save_translation(
    source_text: str,
    target_text: str,
    source_language: str,
    target_language: str,
):
    tid = history_service.save(
        user_id=None,
        source_text=source_text,
        target_text=target_text,
        source_lang=source_language,
        target_lang=target_language,
    )
    return {"status": "ok", "data": {"id": tid}, "error": None}

 
@router.post("/image")
async def translate_static_image(
    image: UploadFile = File(...),
    target_language: str = "ja"
):
    """Translate a single static image (fallback when live overlay fails)."""
    try:
        with record_translation_latency():
            raw = await image.read()
        start = time.perf_counter()
        processed = enhance_contrast(raw)
        ocr_results = await ocr_service.extract_text(processed)
        if not ocr_results:
            empty_payload = {
                "segments": [],
                "source_language": None,
                "target_language": target_language,
                "latency_ms": 0.0,
            }
            return {"status": "ok", "data": empty_payload, "error": None}
        source_lang = detect_language(" ".join(r.text for r in ocr_results))
        segments: list[LiveFrameSegment] = []
        for r in ocr_results:
            tr = await translation_model.translate(
                r.text,
                target_language=target_language,
                source_language=source_lang,
            )
            segment = LiveFrameSegment(
                text=r.text,
                translated=tr["translated_text"],
                x1=r.bbox[0],
                y1=r.bbox[1],
                x2=r.bbox[2],
                y2=r.bbox[3],
                confidence=r.confidence,
            )
            segments.append(segment)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result = LiveFrameResponse(
            segments=segments,
            source_language=source_lang,
            target_language=target_language,
            latency_ms=latency_ms,
        )
        data_payload = result.dict()
        data_payload = mapper.transform_outbound(data_payload)
        deprecated_used = mapper.audit(data_payload)
        if deprecated_used:
            logger.info(
                "deprecated_fields_outbound",
                extra={"fields": deprecated_used},
            )
        return {"status": "ok", "data": data_payload, "error": None}
    except Exception as e:
        err = {
            "status": "error",
            "data": None,
            "error": f"STATIC_IMAGE_TRANSLATION_FAILED: {str(e)}",
        }
        return JSONResponse(status_code=400, content=err)
