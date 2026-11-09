import os
import hashlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.db.models import Dataset, DataRecord, DatasetStatistics, DatasetColumn
from app.services.grounding_validator import GroundingValidator

logger = logging.getLogger(__name__)

class DatasetAnalysisService:
    # In-memory caching
    # Key: dataset_id -> { "dataset_hash": str, "timestamp": datetime, "analysis": Dict }
    _cache: Dict[int, Dict[str, Any]] = {}

    @classmethod
    def get_analysis(cls, db: Session, dataset_id: int, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieves the canonical DatasetAnalysisObject for a dataset.
        Uses cached dictionary if hash matches and force_refresh is False.
        """
        # Load records to compute MD5 payload hash
        records = db.query(DataRecord).filter(DataRecord.dataset_id == dataset_id).order_by(DataRecord.row_index.asc()).all()
        if not records:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            filename = ds.name if ds else f"dataset_{dataset_id}.csv"
            return {
                "dataset_id": dataset_id,
                "filename": filename,
                "row_count": 0,
                "completeness_pct": 0.0,
                "missing_count": 0,
                "duplicate_count": 0,
                "detected_domains": {
                    "primary_domain": "Empty Dataset",
                    "secondary_domain": None,
                    "confidence_score": 0,
                    "detection_reason": "No data records found in database for dataset."
                },
                "schema_metadata": {
                    "all_columns": [],
                    "numerical_measures": [],
                    "categorical_dimensions": [],
                    "date_columns": [],
                    "primary_measure": None,
                    "primary_dimension": None,
                    "primary_date": None
                },
                "kpis": {},
                "kpi_metadata": {},
                "anomaly_summaries": [],
                "recommendations": [],
                "chart_specifications": {
                    "trend_data": [],
                    "category_data": []
                },
                "metadata": {
                    "dataset_hash": "",
                    "analysis_version": "2.0.0",
                    "generated_timestamp": datetime.utcnow().isoformat()
                }
            }

        # Compute payload MD5 hash
        payload_str = "".join([str(r.payload) for r in records])
        dataset_hash = hashlib.md5(payload_str.encode('utf-8')).hexdigest()

        if not force_refresh and dataset_id in cls._cache:
            cache_entry = cls._cache[dataset_id]
            if cache_entry.get("dataset_hash") == dataset_hash:
                logger.info(f"Serving canonical DatasetAnalysisObject from cache for dataset #{dataset_id}")
                return cache_entry["analysis"]

        logger.info(f"Cache miss or stale hash. Executing canonical analysis pipeline for dataset #{dataset_id}")
        analysis_obj = cls._execute_analysis(db, dataset_id, records, dataset_hash)
        cls._cache[dataset_id] = {
            "dataset_hash": dataset_hash,
            "timestamp": datetime.utcnow(),
            "analysis": analysis_obj
        }
        return analysis_obj

    @classmethod
    def invalidate_cache(cls, dataset_id: Optional[int] = None) -> None:
        """
        Invalidates cached analysis.
        """
        if dataset_id is not None:
            if dataset_id in cls._cache:
                del cls._cache[dataset_id]
                logger.info(f"Invalidated analysis cache for dataset #{dataset_id}")
        else:
            cls._cache.clear()
            logger.info("Cleared all analysis caches.")

    @classmethod
    def _execute_analysis(cls, db: Session, dataset_id: int, records: List[DataRecord], dataset_hash: str) -> Dict[str, Any]:
        df = pd.DataFrame([r.payload for r in records])
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        filename = ds.name if ds else f"dataset_{dataset_id}.xlsx"

        # --- Dynamic Columns Classification ---
        # Coerce columns to numeric if they contain numeric-like strings
        for col in df.columns:
            try:
                # Disqualify columns that clearly look like descriptions or string IDs
                if any(k in str(col).lower() for k in ["name", "email", "category", "region", "product", "role", "subject"]):
                    continue
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notna().sum() / len(df) > 0.7:
                    df[col] = coerced
            except Exception:
                pass

        all_cols = df.columns.tolist()
        
        # 1. Date/Time Columns
        date_cols = []
        for col in all_cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            elif 'date' in str(col).lower() or 'time' in str(col).lower():
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() / len(df) > 0.5:
                        date_cols.append(col)
                except Exception:
                    pass

        # 2. Numerical Measures
        numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_measures = []
        for col in numeric_candidates:
            # Disqualify highly unique integer IDs or indexes
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
            if unique_ratio > 0.95 and pd.api.types.is_integer_dtype(df[col]):
                if any(k in str(col).lower() for k in ["id", "key", "index", "pk", "fk", "code", "zip", "phone"]):
                    continue
            if any(k in str(col).lower() for k in ["id", "zip", "code", "index", "phone", "key"]):
                continue
            if df[col].std() == 0:  # No variance
                continue
            numerical_measures.append(col)

        # 3. Categorical Dimensions
        categorical_dimensions = []
        for col in all_cols:
            if col not in numerical_measures and col not in date_cols:
                unique_count = df[col].nunique()
                if 1 < unique_count < 100:
                    unique_ratio = unique_count / len(df) if len(df) > 0 else 1
                    if unique_ratio < 0.9:
                        categorical_dimensions.append(col)

        # --- Domain Detection Confidence Heuristics ---
        primary_domain, secondary_domain, confidence_score, reasons = cls._detect_domains(all_cols, df, numerical_measures, categorical_dimensions)

        # Cache dynamic clean metrics
        for col in numerical_measures:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # --- Dynamic KPI Generation ---
        kpis = {}
        row_count = len(df)
        completeness_pct = float(ds.quality_score) if (ds and ds.quality_score is not None) else 100.0
        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())

        # Set default quality score as KPI
        kpis["dataset_health_score"] = completeness_pct

        if primary_domain == "Student/Education":
            # Total Students
            student_col = next((c for c in all_cols if any(k in c.lower() for k in ["student", "id", "name"])), None)
            if student_col:
                kpis["total_students"] = int(df[student_col].nunique())
            
            # Departments
            dept_col = next((c for c in all_cols if any(k in c.lower() for k in ["dept", "department", "subject", "branch", "class", "course", "major"])), None)
            if dept_col:
                kpis["departments_count"] = int(df[dept_col].nunique())

            # Average Marks
            marks_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["mark", "score", "grade", "gpa"])), None)
            if marks_col:
                kpis["average_marks"] = float(round(df[marks_col].mean(), 2))
                
                # Pass Percentage
                max_val = df[marks_col].max()
                pass_val = 40.0 if max_val > 10 else 2.0  # Assumes 100 scale or GPA 4 scale
                passed = (df[marks_col] >= pass_val).sum()
                kpis["pass_percentage"] = float(round((passed / row_count) * 100.0, 1))

            # Attendance
            att_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["attendance", "present", "att_rate"])), None)
            if att_col:
                kpis["average_attendance"] = float(round(df[att_col].mean(), 1))

        elif primary_domain == "HR/Employee":
            kpis["total_employees"] = row_count
            
            salary_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["salary", "comp", "pay", "wage"])), None)
            if salary_col:
                kpis["average_salary"] = float(round(df[salary_col].mean(), 2))

            dept_col = next((c for c in all_cols if any(k in c.lower() for k in ["dept", "department", "role", "team", "designation"])), None)
            if dept_col:
                kpis["departments_count"] = int(df[dept_col].nunique())

            attrition_col = next((c for c in all_cols if any(k in c.lower() for k in ["attrition", "left", "terminated"])), None)
            if attrition_col:
                yes_count = df[attrition_col].astype(str).str.lower().str.strip().isin(["yes", "1", "true", "y"]).sum()
                kpis["attrition_rate"] = float(round((yes_count / row_count) * 100.0, 1))

        elif primary_domain == "Manufacturing":
            yield_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["yield", "output", "produced", "volume"])), None)
            if yield_col:
                kpis["total_yield"] = float(df[yield_col].sum())

            defect_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["defect", "failed", "rejected"])), None)
            if defect_col and yield_col:
                kpis["defect_rate"] = float(round((df[defect_col].sum() / df[yield_col].sum()) * 100.0, 2)) if df[yield_col].sum() > 0 else 0.0

            machine_col = next((c for c in all_cols if any(k in c.lower() for k in ["machine", "line", "station"])), None)
            if machine_col:
                kpis["machine_count"] = int(df[machine_col].nunique())

            eff_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["efficiency", "performance", "speed"])), None)
            if eff_col:
                kpis["machine_efficiency"] = float(round(df[eff_col].mean(), 1))

        elif primary_domain == "Healthcare":
            kpis["total_patients"] = row_count
            
            stay_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["stay", "days", "duration"])), None)
            if stay_col:
                kpis["average_stay_days"] = float(round(df[stay_col].mean(), 1))

            cost_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["cost", "charge", "bill", "amount"])), None)
            if cost_col:
                kpis["average_treatment_cost"] = float(round(df[cost_col].mean(), 2))

        elif primary_domain == "Sales/Finance":
            rev_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["revenue", "sales", "amount", "price"])), None)
            if rev_col:
                kpis["total_revenue"] = float(df[rev_col].sum())

            profit_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["profit", "earnings", "margin"])), None)
            if profit_col:
                kpis["total_profit"] = float(df[profit_col].sum())

            orders_col = next((c for c in numerical_measures if any(k in c.lower() for k in ["order", "transactions", "quantity"])), None)
            if orders_col:
                kpis["total_orders"] = int(df[orders_col].sum())
            else:
                kpis["total_orders"] = row_count

            cust_col = next((c for c in all_cols if any(k in c.lower() for k in ["customer", "client", "buyer"])), None)
            if cust_col:
                kpis["active_customers"] = int(df[cust_col].nunique())

            if rev_col and profit_col and kpis.get("total_revenue", 0.0) > 0:
                kpis["profit_margin"] = float(round((kpis["total_profit"] / kpis["total_revenue"]) * 100.0, 2))

        else: # Generic/Custom Dataset fallback
            # Construct generic KPIs from first 2 measures and 2 dimensions
            if len(numerical_measures) > 0:
                kpis["primary_measure_sum"] = float(df[numerical_measures[0]].sum())
                kpis["primary_measure_avg"] = float(round(df[numerical_measures[0]].mean(), 2))
            if len(numerical_measures) > 1:
                kpis["secondary_measure_avg"] = float(round(df[numerical_measures[1]].mean(), 2))
            if len(categorical_dimensions) > 0:
                kpis["primary_dimension_groups"] = int(df[categorical_dimensions[0]].nunique())

        # Compile KPI Explainability Metadata
        kpi_metadata = {}
        for kpi, val in kpis.items():
            kpi_metadata[kpi] = {
                "formula": f"Aggregation: sum() or mean() of the matching column mapping.",
                "source_columns": [c for c in all_cols if any(k in c.lower() for k in [kpi, "revenue", "profit", "mark", "gpa", "salary", "yield", "patient"])],
                "rows_used": row_count,
                "missing_values": missing_count,
                "confidence_score": float(completeness_pct)
            }

        # --- Dynamic Visualization Engine ---
        chart_specs = cls._generate_charts(df, numerical_measures, categorical_dimensions, date_cols)

        # --- Anomaly Summaries (IQR based Outliers) ---
        anomalies = []
        if numerical_measures:
            target = numerical_measures[0]
            clean_series = pd.to_numeric(df[target], errors='coerce').dropna()
            if not clean_series.empty and len(clean_series) >= 4:
                q75, q25 = np.percentile(clean_series, [75, 25])
                iqr = q75 - q25
                threshold = q75 + 1.5 * iqr
                outliers = df[pd.to_numeric(df[target], errors='coerce') > threshold]
                m_val = clean_series.mean()
                for idx, r in outliers.head(5).iterrows():
                    anomalies.append({
                        "row_index": int(idx),
                        "measure": target,
                        "value": float(r[target]),
                        "deviation_pct": float(round((r[target] - m_val) / m_val * 100, 1)) if m_val > 0 else 0.0
                    })

        # --- Priority Recommendations (Exactly 5, dynamically bound) ---
        recommendations = []
        recs_count = 0
        
        # Rec 1: completeness
        if completeness_pct < 95:
            recommendations.append({
                "title": "Establish Source Entry Validation Checks",
                "priority": "High",
                "benefit": "Eliminates empty data reporting errors",
                "impact": f"Clears cell gaps for {missing_count} missing entries.",
                "confidence_score": 92.0
            })
            recs_count += 1
        
        # Rec 2: anomaly focus
        if anomalies:
            recommendations.append({
                "title": f"Audit {numerical_measures[0].replace('_',' ').title()} Outliers",
                "priority": "High",
                "benefit": "Stabilizes statistical tracking trends",
                "impact": f"Corrects {len(anomalies)} severe outlier records.",
                "confidence_score": 95.0
            })
            recs_count += 1

        # Rec 3-5: dynamic categorical groupings
        if categorical_dimensions:
            dim_name = categorical_dimensions[0].replace('_',' ').title()
            recommendations.append({
                "title": f"Optimize Distribution across {dim_name}",
                "priority": "Medium",
                "benefit": "Increases segmental output returns",
                "impact": f"Standardizes operations under categorical segments.",
                "confidence_score": 88.0
            })
            recs_count += 1

        if date_cols:
            recommendations.append({
                "title": "Establish Historical Seasonal Baselines",
                "priority": "Medium",
                "benefit": "Prevents period-over-period capacity bottlenecks",
                "impact": "Maps operational shifts across chronological timestamps.",
                "confidence_score": 90.0
            })
            recs_count += 1

        # Fill up to 5 recommendations with distinct items
        fallbacks = [
            {
                "title": "Establish Automated Schema Monitoring Alert triggers",
                "priority": "Low",
                "benefit": "Triggers alert flags immediately upon metric drift",
                "impact": "Keeps operations within calculated safe variance parameters.",
                "confidence_score": 85.0
            },
            {
                "title": "Implement Segmented Quality Score Reports",
                "priority": "Low",
                "benefit": "Identifies localized record errors or missing fields early",
                "impact": "Improves overall database completeness metrics.",
                "confidence_score": 82.0
            },
            {
                "title": "Audit Data Pipeline Logging Infrastructure",
                "priority": "Low",
                "benefit": "Maintains trace history for pipeline errors",
                "impact": "Ensures transaction logging remains consistent.",
                "confidence_score": 80.0
            },
            {
                "title": "Optimize Data Storage Partitioning and Indexing",
                "priority": "Low",
                "benefit": "Improves analytics search times and retrieval latencies",
                "impact": "Reduces query execution load on large tables.",
                "confidence_score": 86.0
            },
            {
                "title": "Set Up Scheduled Backup and Recovery Protocols",
                "priority": "Low",
                "benefit": "Guarantees system continuity in case of failure",
                "impact": "Secures uploaded files and database integrity.",
                "confidence_score": 89.0
            }
        ]

        for fb in fallbacks:
            if len(recommendations) >= 5:
                break
            # Avoid adding if a recommendation with the same title already exists
            if not any(r["title"] == fb["title"] for r in recommendations):
                recommendations.append(fb)

        recommendations = recommendations[:5]

        # Compile final canonical response object
        analysis_obj = {
            "dataset_id": dataset_id,
            "filename": filename,
            "row_count": row_count,
            "completeness_pct": completeness_pct,
            "missing_count": missing_count,
            "duplicate_count": duplicate_count,
            "detected_domains": {
                "primary_domain": primary_domain,
                "secondary_domain": secondary_domain,
                "confidence_score": confidence_score,
                "detection_reason": "; ".join(reasons)
            },
            "schema_metadata": {
                "all_columns": all_cols,
                "numerical_measures": numerical_measures,
                "categorical_dimensions": categorical_dimensions,
                "date_columns": date_cols,
                "primary_measure": numerical_measures[0] if numerical_measures else None,
                "primary_dimension": categorical_dimensions[0] if categorical_dimensions else None,
                "primary_date": date_cols[0] if date_cols else None
            },
            "kpis": kpis,
            "kpi_metadata": kpi_metadata,
            "anomaly_summaries": anomalies,
            "recommendations": recommendations,
            "chart_specifications": chart_specs,
            "metadata": {
                "dataset_hash": dataset_hash,
                "analysis_version": "2.0.0",
                "generated_timestamp": datetime.utcnow().isoformat()
            }
        }

        # --- Debug Logging requirement ---
        logger.info(
            f"Grounding Debug Log - Selected Dataset ID: {dataset_id}, Filename: {filename}\n"
            f"Detected Measures: {numerical_measures}\n"
            f"Detected Dimensions: {categorical_dimensions}\n"
            f"Detected Datetime Fields: {date_cols}\n"
            f"Generated KPIs keys: {list(kpis.keys())}\n"
            f"Retrieval Source: DataRecord Database table\n"
            f"Fallback Logic Used: {'Yes (classified as Generic)' if primary_domain == 'Generic Dataset' else 'No'}"
        )

        return analysis_obj

    @staticmethod
    def _detect_domains(all_cols: List[str], df: pd.DataFrame, measures: List[str], dimensions: List[str]) -> tuple[str, Optional[str], int, List[str]]:
        """
        Evaluates column schema & heuristics to compute domain confidence score.
        """
        scores = {
            "Student/Education": 0,
            "HR/Employee": 0,
            "Manufacturing": 0,
            "Healthcare": 0,
            "Sales/Finance": 0
        }
        reasons = []

        # Semantics / Header checks
        cols_lower = [c.lower() for c in all_cols]

        # Student Keywords
        student_keys = ["student", "mark", "score", "grade", "gpa", "cgpa", "attendance", "course", "subject", "roll"]
        student_matches = [k for k in student_keys if any(k in c for c in cols_lower)]
        if student_matches:
            scores["Student/Education"] += len(student_matches) * 20
            reasons.append(f"Student identifiers detected: {student_matches}")

        # Cell-values Designation Check for Students
        # If the dataset contains designative/role columns and has "student" cell values, it's a strong signal.
        # However, to avoid false positives on Sales datasets with customer demographics, we skip if it has clear Sales metrics.
        looks_like_sales = any(any(k in c.lower() for k in ["revenue", "profit", "sales", "orders"]) for c in all_cols)
        if not looks_like_sales:
            has_student_values = False
            for c in all_cols:
                if any(k in c.lower() for k in ["designation", "role", "title", "occupation", "category", "type", "name"]):
                    try:
                        if df is not None and not df.empty:
                            if df[c].head(50).astype(str).str.lower().str.contains("student").any():
                                has_student_values = True
                                break
                    except Exception:
                        pass
            if has_student_values:
                scores["Student/Education"] += 80
                reasons.append("Row data contains cell values with 'student' designation")

        # HR Keywords
        hr_keys = ["employee", "salary", "hire", "department", "role", "attrition", "performance", "tenure"]
        hr_matches = [k for k in hr_keys if any(k in c for c in cols_lower)]
        if hr_matches:
            scores["HR/Employee"] += len(hr_matches) * 20
            reasons.append(f"HR identifiers detected: {hr_matches}")

        # Manufacturing Keywords
        mfg_keys = ["yield", "defect", "machine", "output", "production", "efficiency", "failure", "part"]
        mfg_matches = [k for k in mfg_keys if any(k in c for c in cols_lower)]
        if mfg_matches:
            scores["Manufacturing"] += len(mfg_matches) * 20
            reasons.append(f"Manufacturing identifiers detected: {mfg_matches}")

        # Healthcare Keywords
        hc_keys = ["patient", "recovery", "diagnosis", "treatment", "admission", "doctor", "hospital", "symptom"]
        hc_matches = [k for k in hc_keys if any(k in c for c in cols_lower)]
        if hc_matches:
            scores["Healthcare"] += len(hc_matches) * 20
            reasons.append(f"Healthcare identifiers detected: {hc_matches}")

        # Sales/Finance Keywords
        sales_keys = ["revenue", "sales", "profit", "margin", "cost", "price", "amount", "customer", "transaction"]
        sales_matches = [k for k in sales_keys if any(k in c for c in cols_lower)]
        if sales_matches:
            scores["Sales/Finance"] += len(sales_matches) * 20
            reasons.append(f"Sales/Finance identifiers detected: {sales_matches}")

        # Calculate best domain
        best_domain = max(scores, key=scores.get)
        confidence = min(scores[best_domain], 100)

        # Multi-domain detection support
        secondary_domain = None
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1][1] >= 40:
            secondary_domain = sorted_scores[1][0]

        if confidence < 60:
            return "Generic Dataset", None, 100, ["Confidence score is below 60%. Classifying as Generic Dataset."]

        return best_domain, secondary_domain, confidence, reasons

    @staticmethod
    def _generate_charts(df: pd.DataFrame, measures: List[str], dimensions: List[str], dates: List[str]) -> Dict[str, Any]:
        """
        Creates recharts specs dynamically based on target measures/dimensions.
        """
        trend_data = []
        category_data = []

        p_measure = measures[0] if measures else None
        p_dim = dimensions[0] if dimensions else None
        p_date = dates[0] if dates else None

        if p_date and p_measure:
            try:
                df_sorted = df.dropna(subset=[p_date]).sort_values(by=p_date)
                df_sorted['period_str'] = pd.to_datetime(df_sorted[p_date], errors='coerce').dt.strftime('%b %y')
                grouped = df_sorted.groupby('period_str', sort=False)[p_measure].sum().reset_index()
                for _, r in grouped.tail(6).iterrows():
                    trend_data.append({
                        "date": str(r["period_str"]),
                        "value": float(round(r[p_measure], 2))
                    })
            except Exception:
                pass

        if p_dim and p_measure:
            try:
                grouped = df.groupby(p_dim)[p_measure].mean().reset_index()
                colors = ['#6366f1', '#a855f7', '#3b82f6', '#10b981', '#f59e0b']
                for idx, r in grouped.head(6).iterrows():
                    category_data.append({
                        "name": str(r[p_dim]),
                        "value": float(round(r[p_measure], 2)),
                        "color": colors[idx % len(colors)]
                    })
            except Exception:
                pass

        # Fallbacks for empty results
        if not trend_data:
            trend_data = []
        if not category_data:
            category_data = []

        return {
            "trend_data": trend_data,
            "category_data": category_data
        }
