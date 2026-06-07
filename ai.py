"""
src/routes/ai.py — AI processing endpoints

Responsibilities (this file only):
  • Validate incoming request (Pydantic)
  • Orchestrate: sanitize → memory fetch → LLM → memory store
  • Return structured response

Does NOT touch: Gmail, DB, credentials, label storage.
"""
import asyncio
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.models import (
    AiProcessing,
    AnalysisResult,
    BatchEmailResult,
    ProcessBatchRequest,
    ProcessBatchResponse,
    ProcessEmailRequest,
    ProcessEmailResponse,
    Summary,
)
from src.services.memory import add_memory, search_memories
from src.services.sanitizer import HTMLSanitizer
from src.services.summarizer import EmailSummarizer

logger = logging.getLogger(__name__)
router = APIRouter()

_summarizer: EmailSummarizer | None = None
_sanitizer = HTMLSanitizer()


def _get_summarizer() -> EmailSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = EmailSummarizer()
    return _summarizer


# ── Single email ──────────────────────────────────────────────────────────────

@router.post(
    "/process-email",
    response_model=ProcessEmailResponse,
    summary="Analyse a single email with AI",
)
async def process_email(payload: ProcessEmailRequest) -> ProcessEmailResponse:
    user_id    = payload.user_id
    email_dict = _payload_to_dict(payload.email)
    subject    = email_dict.get("subject", "No Subject")
    message_id = email_dict["message_id"]

    logger.info("─" * 60)
    logger.info(f"🤖 AI  user={user_id[:20]}  msg={message_id}  subject={subject[:50]}")

    try:
        # 1. Sanitize HTML → clean plain text + preserve raw
        sanitized                 = _sanitizer.sanitize_email(email_dict)
        email_dict["body"]        = sanitized["body"]
        email_dict["raw_content"] = sanitized["raw_content"]

        # 2. Fetch Mem0 memories (non-fatal)
        query    = f"emails about: {subject[:200]}" if subject.strip() else "email processing"
        memories = await _safe_search(user_id, query)

        # 3. LLM analysis
        llm_result = await _get_summarizer().analyse(
            email           = email_dict,
            user_profile    = payload.user_profile,
            existing_labels = payload.existing_labels or [],
            user_memories   = memories,
        )

        # 4. Store memory in background (fire-and-forget)
        asyncio.create_task(_safe_store(user_id, email_dict, llm_result))

        analysis = _build_analysis(llm_result, email_dict)

        logger.info(
            f"✅ Done → category={analysis.summary.category} "
            f"priority={analysis.summary.priority} "
            f"labels={analysis.labels}"
        )

        return ProcessEmailResponse(
            success      = True,
            message_id   = message_id,
            account_id   = email_dict.get("account_id"),
            account_type = email_dict.get("account_type"),
            analysis     = analysis,
        )

    except Exception as e:
        logger.error(f"❌ process_email failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch ─────────────────────────────────────────────────────────────────────

@router.post(
    "/process-emails",
    response_model=ProcessBatchResponse,
    summary="Analyse a batch of emails (max 20)",
)
async def process_emails_batch(payload: ProcessBatchRequest) -> ProcessBatchResponse:
    user_id = payload.user_id
    emails  = payload.emails

    logger.info(f"🤖 BATCH  user={user_id[:20]}  count={len(emails)}")

    subjects = " ".join(e.subject or "" for e in emails[:5])
    query    = f"emails about: {subjects[:200]}" if subjects.strip() else "email processing"
    memories = await _safe_search(user_id, query)

    summarizer = _get_summarizer()
    semaphore  = asyncio.Semaphore(5)

    async def _process_one(email_payload, index: int) -> BatchEmailResult:
        async with semaphore:
            email_dict = _payload_to_dict(email_payload)
            try:
                sanitized                 = _sanitizer.sanitize_email(email_dict)
                email_dict["body"]        = sanitized["body"]
                email_dict["raw_content"] = sanitized["raw_content"]

                llm_result = await summarizer.analyse(
                    email           = email_dict,
                    user_profile    = payload.user_profile,
                    existing_labels = payload.existing_labels or [],
                    user_memories   = memories,
                )

                asyncio.create_task(_safe_store(user_id, email_dict, llm_result))

                analysis = _build_analysis(llm_result, email_dict)

                logger.info(
                    f"  [{index}/{len(emails)}] ✅ {(email_payload.subject or '')[:40]} "
                    f"→ {analysis.summary.category}/{analysis.summary.priority}"
                )

                return BatchEmailResult(
                    success    = True,
                    message_id = email_payload.message_id,
                    account_id = email_dict.get("account_id"),
                    analysis   = analysis,
                )

            except Exception as e:
                logger.error(f"  [{index}/{len(emails)}] ❌ {e}", exc_info=True)
                return BatchEmailResult(
                    success    = False,
                    message_id = email_payload.message_id,
                    error      = str(e),
                )

    results   = list(await asyncio.gather(*[_process_one(e, i) for i, e in enumerate(emails, 1)]))
    succeeded = sum(1 for r in results if r.success)

    logger.info(f"🎉 Batch done: {succeeded}/{len(emails)} succeeded")

    return ProcessBatchResponse(
        total     = len(results),
        succeeded = succeeded,
        failed    = len(results) - succeeded,
        results   = results,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _payload_to_dict(email_payload) -> Dict[str, Any]:
    """Convert EmailPayload → plain dict the services expect."""
    label_ids = email_payload.label_ids or []
    return {
        "message_id":   email_payload.message_id,
        "thread_id":    email_payload.thread_id,
        "subject":      email_payload.subject      or "",
        "from":         email_payload.sender       or "",
        "to":           email_payload.to           or "",
        "cc":           email_payload.cc           or "",
        "bcc":          email_payload.bcc          or "",
        "date":         email_payload.date         or "",
        "body":         email_payload.body         or "",
        "snippet":      email_payload.snippet      or "",
        "label_ids":    label_ids,
        "account_id":   email_payload.account_id,
        "account_type": email_payload.account_type or "gmail",
        "is_sent":      "SENT" in label_ids,
    }


def _build_analysis(llm_result: Dict[str, Any], email_dict: Dict[str, Any]) -> AnalysisResult:
    """
    Assemble the final AnalysisResult from LLM output + email metadata.

    Summary fields use camelCase to match the MongoDB SummarySchema exactly.
    Your friend stores analysis.summary under email.summary with no renaming.
    """
    return AnalysisResult(
        summary = Summary(
            keyPoints      = llm_result.get("key_points", []),
            category       = llm_result.get("category", "others"),
            priority       = llm_result.get("priority", "medium"),
            actionRequired = llm_result.get("action_required", False),
            isActivity     = llm_result.get("is_activity", False),
            actionItems    = llm_result.get("action_items", []),
            events         = llm_result.get("events", []),
            reminders      = llm_result.get("reminders", []),
        ),
        aiProcessing = AiProcessing(
            # processedAt is auto-set to now() by the model default
            model       = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            needsReview = bool(llm_result.get("error")),   # flag fallback cases
            confidence  = None,
        ),
        isSent     = email_dict.get("is_sent", False),
        rawContent = email_dict.get("raw_content", ""),
        labels     = llm_result.get("labels", []),
        tokenUsage = llm_result.get("token_usage"),
    )


async def _safe_search(user_id: str, query: str) -> List[Dict]:
    try:
        return await search_memories(user_id, query)
    except Exception as e:
        logger.warning(f"⚠️ Memory search skipped: {e}")
        return []


async def _safe_store(user_id: str, email: Dict, analysis: Dict) -> None:
    try:
        user_msg = f"Email — Subject: {email.get('subject','')} | From: {email.get('from','')}"
        asst_msg = (
            f"Category: {analysis.get('category')} | "
            f"Priority: {analysis.get('priority')} | "
            f"Key points: {'; '.join(analysis.get('key_points', []))}"
        )
        await add_memory(user_id, user_msg, asst_msg)
    except Exception as e:
        logger.warning(f"⚠️ Memory store skipped: {e}")