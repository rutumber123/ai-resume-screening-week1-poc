"""FastAPI application factory and routes."""

from __future__ import annotations

from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.schemas import HealthResponse, ScreeningResult
from app.services.document_processor import DocumentProcessingError
from app.services.screening import ScreeningService

setup_logging()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI Resume Screening Assistant",
        description="Week 1 GenAI POC — structured resume screening pipeline",
        version=__version__,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            llm_provider=settings.llm_provider,
        )

    @app.post("/api/v1/screen", response_model=ScreeningResult)
    async def screen_resumes(
        job_description: str = Form(..., min_length=1),
        resumes: List[UploadFile] = File(...),
        use_llm: bool = Form(True),
    ) -> ScreeningResult:
        logger.info(
            "Screening request initiated resume_count=%s",
            len(resumes) if resumes else 0,
        )
        if not resumes:
            raise HTTPException(status_code=400, detail="At least one resume is required.")

        payloads = []
        for upload in resumes:
            content = await upload.read()
            payloads.append((upload.filename or "resume.txt", content))

        service = ScreeningService(settings)
        try:
            result = service.screen(job_description, payloads, use_llm=use_llm)
        except DocumentProcessingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected screening failure")
            raise HTTPException(status_code=500, detail="Screening failed.") from exc

        logger.info(
            "Screening completed candidates=%s provider=%s",
            len(result.candidates),
            result.llm_provider,
        )
        return result

    @app.post("/api/v1/screen/text", response_model=ScreeningResult)
    async def screen_text(
        job_description: str = Form(...),
        resume_name: str = Form("candidate.txt"),
        resume_text: str = Form(...),
        use_llm: bool = Form(False),
    ) -> ScreeningResult:
        """Single-resume text screening for demos and evaluation scripts."""
        service = ScreeningService(settings)
        try:
            return service.screen_text_resumes(
                job_description,
                [(resume_name, resume_text)],
                use_llm=use_llm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
