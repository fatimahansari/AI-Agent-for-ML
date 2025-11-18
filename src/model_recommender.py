#python -m src.main examples/customer_churn_metadata.json --context "Need near-real-time scoring within a REST API
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field

DEFAULT_MODEL_NAME = "phi4"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TEMPERATURE = 0.3

class DatasetMetadata(BaseModel):
    """Structured description of a dataset for downstream reasoning."""

    name: str = Field(..., description="Human-friendly dataset name")
    problem_type: Literal[
        "classification",
        "regression",
        "clustering",
        "time-series",
        "nlp",
        "recommendation",
        "anomaly-detection",
    ] = Field(..., description="Primary ML task")
    target: str = Field(..., description="Name of the prediction target column")
    num_samples: Optional[int] = Field(None, description="Approximate number of rows")
    num_features: Optional[int] = Field(None, description="Feature count")
    feature_types: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of feature names to their data types",
    )
    target_distribution: Optional[str] = Field(
        None, description="Summary of how the target variable is distributed"
    )
    missing_values: Optional[str] = Field(
        None, description="Notes about missing data patterns or imputation"
    )
    leakage_risks: Optional[str] = Field(
        None, description="Potential sources of data leakage to avoid"
    )
    text_fields: List[str] = Field(
        default_factory=list, description="Columns that contain free text"
    )
    time_index: Optional[str] = Field(
        None, description="Name of the timestamp column, if any"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Business, latency, interpretability, or resource constraints",
    )
    evaluation_metric: Optional[str] = Field(
        None, description="Preferred metric such as accuracy, ROC-AUC, RMSE, etc."
    )
    domain: Optional[str] = Field(None, description="Application domain context")
    notes: Optional[str] = Field(None, description="Any additional free-form notes")


class ModelRecommenderAgent:
    """LangChain-powered agent that suggests ML models from dataset metadata."""

    def __init__(self) -> None:
        self.llm = OllamaLLM(
            model=DEFAULT_MODEL_NAME,
            base_url=DEFAULT_BASE_URL,
            temperature=DEFAULT_TEMPERATURE,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an ML solution architect. Given dataset metadata, return 3-5
                    concrete model families or algorithms that best fit, explaining why,
                    what assumptions must hold, required preprocessing, and evaluation
                    considerations. Reference constraints and risks explicitly.
                    
                    CRITICAL COMPATIBILITY REQUIREMENTS - Only recommend compatible models:
                    
                    Classification problems:
                    - ALL classification models require at least 2 unique classes in target
                    - SVM: Requires >1 class (fails if only 1 class present)
                    - Logistic Regression: Requires at least 2 classes for multi-class
                    - Random Forest, XGBoost, LightGBM: Work with 2+ classes
                    - Neural Networks: Need at least 2 classes, preferably more data per class
                    
                    Dataset size:
                    - Neural Networks: Require larger datasets (1000+ samples recommended)
                    - SVM: Can be slow with large datasets (>10k samples)
                    - Tree-based models: Generally work well with medium datasets
                    
                    IMPORTANT: Check target distribution in metadata - if it shows only 1 class
                    or very few samples per class, avoid recommending models that will fail.
                    Do NOT recommend models that cannot work with the dataset characteristics.
                    
                    Output in concise markdown with numbered recommendations.
                    """,
                ),
                (
                    "human",
                    """Dataset metadata (JSON):
{metadata_json}

Extra context:
{context}

Compose your recommendations.
""",
                ),
            ]
        )
        self.chain: RunnableSequence = self.prompt | self.llm | StrOutputParser()

    def recommend(self, *, metadata: DatasetMetadata, context: Optional[str] = None) -> str:
        """Generate tailored model recommendations."""

        payload = {
            "metadata_json": metadata.model_dump_json(indent=2),
            "context": context or "(no additional context)",
        }
        return self.chain.invoke(payload)
