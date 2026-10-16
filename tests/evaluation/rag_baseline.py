"""
Isolated Baseline Adapter mirroring production RAGPipelineService.query_rag_assistant.
Strictly isolated: does not alter production files.
Enforces strict dataset isolation: raises error if dataset_key is missing.
"""
import time
import pandas as pd
from typing import Dict, Any, List


class ProductionRAGBaselineAdapter:
    """
    Executes the exact logic of production RAGPipelineService.query_rag_assistant
    directly against the isolated dataset dataframe.
    """
    def __init__(self, corpus):
        self.corpus = corpus

    def query(self, query: str, dataset_key: str) -> Dict[str, Any]:
        """
        Execute query using current production logic.
        Enforces strict dataset isolation.
        """
        if not dataset_key:
            raise ValueError("Dataset key is missing. Dataset isolation rule: never select another dataset silently.")

        start_time = time.perf_counter()
        df = self.corpus.get_dataframe(dataset_key)

        # Domain classification
        if dataset_key in ["sample_business_dataset", "enterprise_sales"]:
            domain = "Sales/Finance"
        elif dataset_key == "student_data":
            domain = "Student/Education"
        else:
            domain = "Generic Dataset"

        q_lower = query.lower()
        answer = ""
        sources = [f"Dataset: {dataset_key}", "Grounding Validator Checked"]
        retrieved_indices: List[int] = []  # Production baseline performs NO document ranking

        # 1. Summary branch
        if any(k in q_lower for k in ["what is present", "columns", "summary", "what dataset"]):
            cols_str = ", ".join(df.columns)
            answer = (
                f"The active dataset '{dataset_key}' is classified as a **{domain}** dataset.\n"
                f"- Row count: {len(df)} total records\n"
                f"- Columns detected: {cols_str}"
            )

        # 2. Sales / Finance branch
        elif any(k in q_lower for k in ["revenue", "profit", "sales", "arr"]):
            if domain != "Sales/Finance":
                answer = "The uploaded dataset does not contain revenue information."
            else:
                total_rev = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
                total_prof = float(df["profit"].sum()) if "profit" in df.columns else 0.0
                margin = round((total_prof / total_rev * 100), 2) if total_rev > 0 else 0.0
                answer = (
                    f"Based on the **Sales/Finance** dataset context, total tracked revenue reaches "
                    f"**${total_rev:,.2f}** with aggregate net profit at **${total_prof:,.2f}** "
                    f"(gross margin: **{margin}%**)."
                )

        # 3. Student / Education branch
        elif any(k in q_lower for k in ["marks", "gpa", "score", "attendance", "student"]):
            if domain != "Student/Education":
                answer = "The uploaded dataset does not contain student marks/GPA information."
            else:
                total_st = len(df)
                # In current codebase, if columns do not exist, it abstains or reports defaults
                if "marks" in df.columns or "gpa" in df.columns:
                    score_col = "marks" if "marks" in df.columns else "gpa"
                    avg_score = float(df[score_col].mean())
                    answer = f"Based on the Student/Education dataset context, we track {total_st} students with an average score of {avg_score:.2f}."
                else:
                    answer = "The uploaded dataset does not contain student marks/GPA information."

        # 4. Dynamic Column match & Pandas aggregations
        else:
            matched_col = None
            for col in df.columns:
                if col.lower() in q_lower:
                    matched_col = col
                    break

            if matched_col and pd.api.types.is_numeric_dtype(df[matched_col]):
                if any(k in q_lower for k in ["average", "mean", "avg"]):
                    avg_val = float(df[matched_col].mean())
                    answer = f"The computed average for numeric column '{matched_col}' is **{avg_val:.2f}**."
                else:
                    sum_val = float(df[matched_col].sum())
                    answer = f"The aggregate sum for numeric column '{matched_col}' is **{sum_val:,.2f}**."
            elif matched_col:
                cardinality = int(df[matched_col].nunique())
                top_val = df[matched_col].value_counts().index[0] if len(df) > 0 else "N/A"
                answer = f"Categorical column '{matched_col}' contains **{cardinality}** unique groups. The most frequent category is **'{top_val}'**."
            else:
                answer = "The uploaded dataset does not contain sufficient information to answer this question."

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": query,
            "dataset_key": dataset_key,
            "answer": answer,
            "sources": sources,
            "retrieved_indices": retrieved_indices,
            "latency_ms": elapsed_ms
        }
