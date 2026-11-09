from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User, Insight, AnomalyResult
from app.schemas.schemas import (
    ChatMessage, ChatResponse, InsightResponse, AnomalyResponse,
    KPIDashboardResponse, ChartsDashboardResponse, BusinessHealthResponse, DashboardSummaryResponse
)
from app.api.deps import get_current_user, check_dataset_access
from app.services.rag_service import RAGChatService
from app.services.insights_service import InsightsEngineService
from app.services.anomaly_service import AnomalyDetectionService
from app.services.analytics_service import AnalyticsDashboardService
from app.services.dataset_analysis_service import DatasetAnalysisService


router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def rag_chat(
    chat_in: ChatMessage,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not chat_in.dataset_id:
        raise HTTPException(
            status_code=400,
            detail="Active dataset_id is required. Please select an active dataset."
        )

    check_dataset_access(db, current_user, chat_in.dataset_id)

    response = RAGChatService.query_rag_assistant(
        db=db,
        message=chat_in.message,
        dataset_id=chat_in.dataset_id
    )
    return response

@router.get("/insights/{dataset_id}", response_model=List[InsightResponse])
def get_insights(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_dataset_access(db, current_user, dataset_id)
    insights = db.query(Insight).filter(Insight.dataset_id == dataset_id).all()
    if not insights:
        insights = InsightsEngineService.generate_all_insights(db, dataset_id)
    return insights

@router.get("/anomalies/{dataset_id}", response_model=List[AnomalyResponse])
def get_anomalies(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_dataset_access(db, current_user, dataset_id)
    anomalies = db.query(AnomalyResult).filter(AnomalyResult.dataset_id == dataset_id).all()
    if not anomalies:
        anomalies = AnomalyDetectionService.detect_anomalies(db, dataset_id)
    return anomalies

# Phase 4 Analytics Dashboard Routes
@router.get("/dashboard", response_model=DashboardSummaryResponse)
def get_dashboard_summary(
    dataset_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    product: Optional[str] = None,
    region: Optional[str] = None,
    category: Optional[str] = None,
    customer: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_dataset_access(db, current_user, dataset_id)
    analysis = DatasetAnalysisService.get_analysis(db, dataset_id)
    domain = analysis["detected_domains"]["primary_domain"]
    
    if domain == "Student/Education":
        title = "Student Performance Analytics"
        subtitle = "Detailed academic metrics and student evaluation metrics"
    elif domain == "HR/Employee":
        title = "Workforce Analytics"
        subtitle = "Detailed employee statistics, performance metrics, and attrition tracking"
    elif domain == "Manufacturing":
        title = "Manufacturing Performance Analytics"
        subtitle = "Detailed production yield, defect rates, and machine performance analytics"
    elif domain == "Sales/Finance":
        title = "Sales Performance Analytics"
        subtitle = "Detailed revenue margins, profit figures, and orders performance metrics"
    else:
        title = "Dataset Analytics Workspace"
        subtitle = "General statistical overview and data distributions"

    # Filter metadata mapping (maps frontend filter keys to friendly dataset-specific names)
    filter_metadata = {}
    schema = analysis.get("schema_metadata", {})
    all_cols = schema.get("all_columns", [])
    
    product_col = next((c for c in all_cols if any(k in c.lower() for k in ["product", "course", "subject", "major", "item", "part"])), None)
    if product_col:
        filter_metadata["product"] = product_col.replace('_', ' ').title()
        
    region_col = next((c for c in all_cols if any(k in c.lower() for k in ["region", "location", "class", "grade", "branch", "office", "site"])), None)
    if region_col:
        filter_metadata["region"] = region_col.replace('_', ' ').title()
        
    category_col = next((c for c in all_cols if any(k in c.lower() for k in ["category", "gender", "dept", "department", "type", "segment"])), None)
    if category_col:
        filter_metadata["category"] = category_col.replace('_', ' ').title()
        
    customer_col = next((c for c in all_cols if any(k in c.lower() for k in ["customer", "client", "buyer", "student", "employee", "name"])), None)
    if customer_col:
        filter_metadata["customer"] = customer_col.replace('_', ' ').title()

    date_col = next((c for c in all_cols if any(k in c.lower() for k in ["date", "time"]) or c in schema.get("date_columns", [])), None)
    if date_col:
        filter_metadata["date"] = "Date"

    available_filters = list(filter_metadata.keys())

    kpis = AnalyticsDashboardService.get_kpis(
        db, dataset_id, start_date, end_date, product, region, category, customer
    )
    charts = AnalyticsDashboardService.get_charts(
        db, dataset_id, start_date, end_date, product, region, category, customer
    )
    health = AnalyticsDashboardService.get_health_score(db, dataset_id)
    
    return {
        "kpis": kpis,
        "charts": charts,
        "health": health,
        "domain": domain,
        "title": title,
        "subtitle": subtitle,
        "available_filters": available_filters,
        "filter_metadata": filter_metadata
    }

@router.get("/kpis", response_model=KPIDashboardResponse)
def get_dashboard_kpis(
    dataset_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    product: Optional[str] = None,
    region: Optional[str] = None,
    category: Optional[str] = None,
    customer: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_dataset_access(db, current_user, dataset_id)
    return AnalyticsDashboardService.get_kpis(
        db, dataset_id, start_date, end_date, product, region, category, customer
    )

@router.get("/charts", response_model=ChartsDashboardResponse)
def get_dashboard_charts(
    dataset_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    product: Optional[str] = None,
    region: Optional[str] = None,
    category: Optional[str] = None,
    customer: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_dataset_access(db, current_user, dataset_id)
    return AnalyticsDashboardService.get_charts(
        db, dataset_id, start_date, end_date, product, region, category, customer
    )

@router.get("/health", response_model=BusinessHealthResponse)
def get_business_health(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_dataset_access(db, current_user, dataset_id)
    return AnalyticsDashboardService.get_health_score(db, dataset_id)

