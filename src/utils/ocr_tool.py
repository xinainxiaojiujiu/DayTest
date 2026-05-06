import logging
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)

_ocr_engine: Optional[Any] = None


def _get_ocr_engine() -> Any:
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang=settings.OCR_LANG,
            use_gpu=settings.OCR_USE_GPU,
            show_log=False,
        )
        logger.info("PaddleOCR engine initialized | lang=%s | gpu=%s", settings.OCR_LANG, settings.OCR_USE_GPU)
    return _ocr_engine


async def recognize_text(image_path: str) -> str:
    try:
        ocr = _get_ocr_engine()
        result: List = ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""

        texts: List[str] = []
        for line in result[0]:
            if line and len(line) >= 2:
                text: str = line[1][0]
                confidence: float = line[1][1]
                if confidence > 0.5:
                    texts.append(text)

        joined = " ".join(texts)
        logger.info("OCR result | image=%s | text_len=%d", image_path, len(joined))
        return joined

    except Exception as exc:
        logger.error("OCR failed | image=%s | error=%s", image_path, str(exc))
        return ""


async def recognize_defect_image(image_path: str) -> Dict[str, Any]:
    text = await recognize_text(image_path)
    return {
        "image_path": image_path,
        "ocr_text": text,
        "has_text": len(text.strip()) > 0,
    }
