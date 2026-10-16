"""
Dataset document indexer for isolated RAG evaluation.
Loads real CSV files from backend/uploads/cleaned/ and serializes rows into structured documents.
"""
import os
import pandas as pd
from typing import Dict, List, Any


DATASET_PATHS = {
    "sample_business_dataset": os.path.join(
        os.path.dirname(__file__), "..", "..", "backend", "uploads", "cleaned", "cleaned_3_sample_business_dataset.csv"
    ),
    "enterprise_sales": os.path.join(
        os.path.dirname(__file__), "..", "..", "backend", "uploads", "cleaned", "cleaned_23_Enterprise_Sales_2026.csv"
    ),
    "student_data": os.path.join(
        os.path.dirname(__file__), "..", "..", "backend", "uploads", "cleaned", "cleaned_3_Student_data_50_records.csv"
    )
}


class Document:
    def __init__(self, doc_id: str, dataset_key: str, row_idx: int, raw_dict: Dict[str, Any], text: str):
        self.doc_id = doc_id
        self.dataset_key = dataset_key
        self.row_idx = row_idx
        self.raw_dict = raw_dict
        self.text = text

    def __repr__(self):
        return f"<Doc {self.doc_id}: {self.text[:60]}...>"


class DatasetCorpus:
    def __init__(self):
        self.datasets: Dict[str, List[Document]] = {}
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self._load_datasets()

    def _load_datasets(self):
        for key, rel_path in DATASET_PATHS.items():
            abs_path = os.path.abspath(rel_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Required benchmark dataset not found at: {abs_path}")
            
            df = pd.read_csv(abs_path)
            self.dataframes[key] = df
            docs = []
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                parts = [f"Record #{idx}"]
                for col, val in row_dict.items():
                    parts.append(f"{col}: {val}")
                doc_text = " | ".join(parts)
                doc = Document(
                    doc_id=f"{key}_row_{idx}",
                    dataset_key=key,
                    row_idx=idx,
                    raw_dict=row_dict,
                    text=doc_text
                )
                docs.append(doc)
            self.datasets[key] = docs

    def get_documents(self, dataset_key: str) -> List[Document]:
        """Strict dataset isolation: returns documents for the requested dataset only."""
        if not dataset_key or dataset_key not in self.datasets:
            raise KeyError(f"Dataset key '{dataset_key}' is invalid or missing. Dataset isolation enforced.")
        return self.datasets[dataset_key]

    def get_dataframe(self, dataset_key: str) -> pd.DataFrame:
        """Strict dataset isolation: returns DataFrame for the requested dataset only."""
        if not dataset_key or dataset_key not in self.dataframes:
            raise KeyError(f"Dataset key '{dataset_key}' is invalid or missing. Dataset isolation enforced.")
        return self.dataframes[dataset_key].copy()
