"""
src/models.py — All Pydantic request & response models

Ownership map (AI route vs friend's serverless backend):

  AI ROUTE GENERATES:
    summary.keyPoints, summary.category, summary.priority,
    summary.actionRequired, summary.isActivity,
    summary.actionItems, summary.events, summary.reminders,
    labels (strings — friend resolves to ObjectIds),
    isSent (derived from label_ids),
    rawContent (sanitized HTML → plain text),
    aiProcessing { processedAt, model, needsReview }

  FRIEND SENDS (and stores himself):
    userId, accountId, accountType, messageId, threadId,
    sender, to, cc, bcc, subject, date, body, snippet,
    label_ids, draftId, isRead, isStarred, isSummarized,
    responseTime, timeSaved, size, attachments,
    archivedDate, deletedDate, spamReason,
    isDraft, isTrash, isArchived, isErrored, errorReason,
    isReplyTo, references, isReply, isForward,
    summaryFeedback, isResolved, resolvedAt, resolvedType
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# INBOUND  (friend → AI route)
# ─────────────────────────────────────────────────────────────────────────────

class EmailPayload(BaseModel):
    """
    A single email already fetched by the serverless backend.
    Field names mirror raw Gmail / IMAP responses — no renaming needed.
    """

    # Required
    message_id: str = Field(...,  description="Gmail message ID or IMAP UID")

    # Core headers
    thread_id: Optional[str] = Field(None, description="Gmail thread ID (null for IMAP)")
    subject:   Optional[str] = Field("",   description="Subject line")

    # 'from' is a Python keyword — aliased so JSON key stays 'from'
    sender: Optional[str] = Field(
        "", alias="from",
        description="Sender — 'Display Name <email@example.com>' or plain address"
    )
    to:      Optional[str] = Field("",  description="To recipients, comma-separated")
    cc:      Optional[str] = Field("",  description="CC recipients, comma-separated")
    bcc:     Optional[str] = Field("",  description="BCC recipients, comma-separated")
    date:    Optional[str] = Field("",  description="RFC-2822 or ISO date string")
    body:    Optional[str] = Field("",  description="Full body — HTML preferred, plain text OK")
    snippet: Optional[str] = Field("",  description="Short preview — fallback when body is empty")
    label_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="Gmail system labels e.g. ['INBOX', 'SENT', 'UNREAD']. "
                    "'SENT' is used to derive isSent automatically."
    )

    # Account metadata
    account_id: Optional[str] = Field(
        None, description="Connected account address e.g. 'user@gmail.com'"
    )
    account_type: Optional[Literal["gmail", "imap", "outlook", "icloud", "work", "other"]] = Field(
        "gmail", description="Mail provider"
    )

    model_config = {"populate_by_name": True}


class ProcessEmailRequest(BaseModel):
    """POST /api/v1/ai/process-email"""
    user_id: str = Field(..., description="Kinde / internal user ID — scopes Mem0 memory")
    email: EmailPayload
    user_profile: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Pre-fetched user profile. Optional — LLM degrades gracefully without it. "
            "Keys: firstName, lastName, jobTitle, department, company, industry, "
            "interests, priorities, communicationStyle, location, bio, "
            "inboxGoal, communicationFocus, accountCount."
        )
    )
    existing_labels: Optional[List[str]] = Field(
        default_factory=list,
        description="Current label path strings e.g. ['Work', 'Work/Project Phoenix']. "
                    "LLM reuses these before inventing new ones."
    )


class ProcessBatchRequest(BaseModel):
    """POST /api/v1/ai/process-emails — up to 20 emails per call"""
    user_id: str = Field(..., description="Kinde / internal user ID")
    emails: List[EmailPayload] = Field(..., min_length=1, max_length=20)
    user_profile: Optional[Dict[str, Any]] = None
    existing_labels: Optional[List[str]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# OUTBOUND  (AI route → friend)
# Naming convention: camelCase to match MongoDB schema exactly.
# Friend stores analysis.summary as-is; no renaming needed.
# ─────────────────────────────────────────────────────────────────────────────

class ActionItem(BaseModel):
    """Matches ActionItemSchema in MongoDB"""
    description: str
    completed: bool = False
    priority: Literal["high", "normal", "low"] = "normal"


class Event(BaseModel):
    """Matches EventSchema in MongoDB"""
    title:       Optional[str]  = None
    start:       Optional[str]  = None   # ISO datetime string or null
    end:         Optional[str]  = None
    location:    Optional[str]  = None
    description: Optional[str]  = None
    allDay:      bool           = False
    priority:    Literal["high", "normal", "low"] = "normal"


class Reminder(BaseModel):
    """Matches ReminderSchema in MongoDB"""
    title:    Optional[str] = None
    dueDate:  Optional[str] = None   # ISO datetime string or null
    priority: Literal["high", "normal", "low"] = "normal"


class Summary(BaseModel):
    """
    Matches SummarySchema in MongoDB exactly (camelCase).
    Friend stores this object under email.summary — no field renaming needed.
    """
    keyPoints:      List[str]       = Field(default_factory=list)
    category:       Literal["professional", "finance", "marketing", "others"] = "others"
    priority:       Literal["high", "medium", "low"] = "medium"
    actionRequired: bool            = False
    isActivity:     bool            = False
    actionItems:    List[ActionItem] = Field(default_factory=list)
    events:         List[Event]      = Field(default_factory=list)
    reminders:      List[Reminder]   = Field(default_factory=list)


class AiProcessing(BaseModel):
    """
    Matches aiProcessing sub-document in MongoDB.
    Friend stores this under email.aiProcessing.
    """
    processedAt: str  = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of when the AI processed this email"
    )
    model: str        = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    needsReview: bool = False
    confidence:  Optional[float] = Field(
        None, description="0.0–1.0 — reserved for future use, null for now"
    )


class AnalysisResult(BaseModel):
    """
    Everything the AI route generates for one email.
    Split into three groups so your friend knows exactly where each piece goes:

      summary      → store as email.summary
      aiProcessing → store as email.aiProcessing
      isSent       → use to set email.isSent and email.isRead
      rawContent   → store as email.rawContent
      labels       → resolve to ObjectIds, store as email.labels
      tokenUsage   → log / discard (not stored in DB)
    """

    # ── Stored as email.summary ───────────────────────────────────────────────
    summary: Summary

    # ── Stored as email.aiProcessing ─────────────────────────────────────────
    aiProcessing: AiProcessing = Field(default_factory=AiProcessing)

    # ── Stored as email.isSent + email.isRead ────────────────────────────────
    isSent: bool = Field(
        False,
        description="Derived from 'SENT' in label_ids. "
                    "Friend sets isSent=this and isRead=this."
    )

    # ── Stored as email.rawContent ────────────────────────────────────────────
    rawContent: str = Field(
        "",
        description="Sanitized plain-text version of the HTML body. "
                    "Store as email.rawContent."
    )

    # ── Used by friend to resolve email.labels (ObjectIds) ───────────────────
    labels: List[str] = Field(
        default_factory=list,
        description="Label path strings e.g. ['Work/Project Phoenix', 'Newsletter']. "
                    "Friend calls POST /api/tools/labels to resolve these to ObjectIds."
    )

    # ── Internal metadata — log/discard, not stored in DB ────────────────────
    tokenUsage: Optional[Dict[str, Any]] = Field(
        None,
        description="LLM token counts. Not stored in DB — for logging/billing only."
    )


class ProcessEmailResponse(BaseModel):
    """Response for POST /api/v1/ai/process-email"""
    success:      bool
    message_id:   str
    account_id:   Optional[str] = None
    account_type: Optional[str] = None
    analysis:     AnalysisResult


class BatchEmailResult(BaseModel):
    success:    bool
    message_id: str
    account_id: Optional[str]       = None
    analysis:   Optional[AnalysisResult] = None
    error:      Optional[str]        = None


class ProcessBatchResponse(BaseModel):
    total:     int
    succeeded: int
    failed:    int
    results:   List[BatchEmailResult]