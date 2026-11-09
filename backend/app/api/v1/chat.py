from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Conversation, ChatMessageModel, AISettings, User, Dataset
from app.schemas.schemas import (
    ConversationCreate, ConversationUpdate, ConversationResponse, 
    ChatMessageResponse, AISettingsUpdate, AISettingsResponse, ChatMessage
)
from app.api.deps import get_current_user, require_roles, check_dataset_access
from app.services.rag_service import RAGPipelineService

from app.services.dataset_analysis_service import DatasetAnalysisService

router = APIRouter()

@router.post("/conversations", response_model=ConversationResponse)
def create_conversation(
    conv_in: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Validate dataset existence and user permission if a dataset_id is explicitly provided
    if conv_in.dataset_id is not None:
        check_dataset_access(db, current_user, conv_in.dataset_id)

    conv = Conversation(
        user_id=current_user.id,
        dataset_id=conv_in.dataset_id,
        title=conv_in.title or "New Business Intelligence Query",
        model_provider=conv_in.model_provider or "llama3.1"
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    # Dynamic Welcome message and suggestions (No unsafe fallback to another dataset)
    dataset_id = conv.dataset_id
    domain = "Generic"
    suggestions = ["What columns are available?", "Summarize this dataset", "Show key statistics", "Detect anomalies"]

    if dataset_id:
        try:
            analysis = DatasetAnalysisService.get_analysis(db, dataset_id)
            domain = analysis["detected_domains"]["primary_domain"]
            if domain == "Student/Education":
                suggestions = [
                    "Summarize this dataset",
                    "Which students scored highest?",
                    "Show attendance statistics",
                    "Explain the GPA distribution"
                ]
            elif domain == "Sales/Finance":
                suggestions = [
                    "Which products generated the highest sales?",
                    "Show monthly sales trend",
                    "What are the biggest revenue drivers?"
                ]
            elif domain == "HR/Employee":
                suggestions = [
                    "Summarize this dataset",
                    "What is the average employee salary?",
                    "Show attrition rates",
                    "Explain department distributions"
                ]
        except Exception:
            pass

    welcome_msg = ChatMessageModel(
        conversation_id=conv.id,
        sender="assistant",
        message_text=f"Welcome! I am your AI Business Intelligence Assistant powered by LangChain, FAISS RAG, and `{conv.model_provider.upper()}`. Ask any natural language question regarding your active dataset.",
        sources_json=["RAG Vector Storage Active"],
        data_summary_json={"Model Provider": conv.model_provider.upper(), "Status": "Ready for Query"},
        confidence_score=99.0,
        follow_ups_json=suggestions
    )
    db.add(welcome_msg)
    db.commit()
    db.refresh(conv)
    return conv

@router.get("/conversations", response_model=List[ConversationResponse])
def list_conversations(
    search: Optional[str] = None,
    is_pinned: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Conversation).filter(Conversation.user_id == current_user.id)
    if search:
        query = query.filter(Conversation.title.ilike(f"%{search}%"))
    if is_pinned is not None:
        query = query.filter(Conversation.is_pinned == is_pinned)

    conversations = query.order_by(Conversation.is_pinned.desc(), Conversation.updated_at.desc()).all()
    if not conversations:
        # Create default initial conversation
        conv = Conversation(
            user_id=current_user.id,
            title="Enterprise Executive Briefing",
            model_provider="llama3.1"
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)

        welcome_msg = ChatMessageModel(
            conversation_id=conv.id,
            sender="assistant",
            message_text="Welcome! Ask me any question using your uploaded company datasets.",
            sources_json=["RAG Vector Storage Active"],
            data_summary_json={"Status": "Active"},
            confidence_score=99.0
        )
        db.add(welcome_msg)
        db.commit()
        db.refresh(conv)
        conversations = [conv]

    return conversations

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation_history(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Conversation).filter(Conversation.id == conversation_id)
    if current_user.role != "Admin":
        query = query.filter(Conversation.user_id == current_user.id)
    conv = query.first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    # Validate access to the conversation's active dataset if present
    if conv.dataset_id:
        check_dataset_access(db, current_user, conv.dataset_id)

    return conv

@router.put("/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: int,
    conv_in: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Conversation).filter(Conversation.id == conversation_id)
    if current_user.role != "Admin":
        query = query.filter(Conversation.user_id == current_user.id)
    conv = query.first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")


    if conv_in.title is not None:
        conv.title = conv_in.title
    if conv_in.is_pinned is not None:
        conv.is_pinned = conv_in.is_pinned
    if conv_in.model_provider is not None:
        conv.model_provider = conv_in.model_provider

    db.commit()
    db.refresh(conv)
    return conv

@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "Manager", "Analyst"]))
):
    query = db.query(Conversation).filter(Conversation.id == conversation_id)
    if current_user.role != "Admin":
        query = query.filter(Conversation.user_id == current_user.id)
    conv = query.first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")


    db.delete(conv)
    db.commit()
    return {"message": "Conversation successfully deleted."}

@router.post("/conversations/{conversation_id}/messages", response_model=ChatMessageResponse)
def send_rag_message(
    conversation_id: int,
    chat_in: ChatMessage,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Conversation).filter(Conversation.id == conversation_id)
    if current_user.role != "Admin":
        query = query.filter(Conversation.user_id == current_user.id)
    conv = query.first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    dataset_id = chat_in.dataset_id or conv.dataset_id
    if not dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Active dataset_id is required. Please select or provide a dataset."
        )

    # Validate dataset existence and permission before querying (no silent switching)
    check_dataset_access(db, current_user, dataset_id)

    if chat_in.dataset_id and conv.dataset_id != chat_in.dataset_id:
        conv.dataset_id = chat_in.dataset_id
        db.commit()

    assistant_response = RAGPipelineService.execute_rag_query(
        db=db,
        user_id=current_user.id,
        conversation_id=conversation_id,
        user_message=chat_in.message,
        dataset_id=dataset_id,
        model_provider=conv.model_provider
    )
    return assistant_response

@router.get("/settings", response_model=AISettingsResponse)
def get_ai_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    settings_entry = db.query(AISettings).filter(AISettings.user_id == current_user.id).first()
    if not settings_entry:
        settings_entry = AISettings(
            user_id=current_user.id,
            preferred_llm="llama3.1",
            temperature=0.2,
            max_tokens=2048,
            prompt_style="Executive Analytical",
            language="en"
        )
        db.add(settings_entry)
        db.commit()
        db.refresh(settings_entry)
    return settings_entry

@router.put("/settings", response_model=AISettingsResponse)
def update_ai_settings(
    settings_in: AISettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    settings_entry = db.query(AISettings).filter(AISettings.user_id == current_user.id).first()
    if not settings_entry:
        settings_entry = AISettings(user_id=current_user.id)
        db.add(settings_entry)

    if settings_in.preferred_llm:
        settings_entry.preferred_llm = settings_in.preferred_llm
    if settings_in.temperature is not None:
        settings_entry.temperature = settings_in.temperature
    if settings_in.max_tokens is not None:
        settings_entry.max_tokens = settings_in.max_tokens
    if settings_in.prompt_style:
        settings_entry.prompt_style = settings_in.prompt_style
    if settings_in.language:
        settings_entry.language = settings_in.language

    db.commit()
    db.refresh(settings_entry)
    return settings_entry
