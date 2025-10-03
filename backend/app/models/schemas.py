"""
NeuralVerse AI - Pydantic Models and Schemas
Data models for all AI services and features
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

# ==================== ENUMS ====================

class ModelType(str, Enum):
    """AI Model Types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    GENERATIVE = "generative"
    REINFORCEMENT = "reinforcement"
    TIME_SERIES = "time_series"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    AUDIO = "audio"

class Framework(str, Enum):
    """ML Frameworks"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"
    KERAS = "keras"
    HUGGINGFACE = "huggingface"

class TaskType(str, Enum):
    """AI Task Types"""
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    CODE_GENERATION = "code_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    OBJECT_DETECTION = "object_detection"
    SPEECH_RECOGNITION = "speech_recognition"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"

# ==================== CORE MODELS ====================

class ModelBase(BaseModel):
    """Base model for AI models"""
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    model_type: ModelType = Field(..., description="Type of model")
    framework: Framework = Field(..., description="ML framework used")
    version: str = Field(default="1.0.0", description="Model version")
    tags: List[str] = Field(default=[], description="Model tags")
    hyperparameters: Dict[str, Any] = Field(default={}, description="Model hyperparameters")

class ModelCreate(ModelBase):
    """Model creation schema"""
    training_data_path: Optional[str] = Field(None, description="Path to training data")
    validation_data_path: Optional[str] = Field(None, description="Path to validation data")

class ModelResponse(ModelBase):
    """Model response schema"""
    id: str = Field(..., description="Model ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: str = Field(..., description="Model status")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    size_mb: Optional[float] = Field(None, description="Model size in MB")

# ==================== TRAINING MODELS ====================

class TrainingJobBase(BaseModel):
    """Base training job schema"""
    model_id: str = Field(..., description="Model ID to train")
    dataset_config: Dict[str, Any] = Field(..., description="Dataset configuration")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    validation_config: Dict[str, Any] = Field(default={}, description="Validation configuration")

class TrainingJobCreate(TrainingJobBase):
    """Training job creation schema"""
    pass

class TrainingJobResponse(TrainingJobBase):
    """Training job response schema"""
    id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Training status")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Training start time")
    completed_at: Optional[datetime] = Field(None, description="Training completion time")
    metrics: Dict[str, Any] = Field(default={}, description="Training metrics")
    logs: List[str] = Field(default=[], description="Training logs")

# ==================== INFERENCE MODELS ====================

class InferenceRequest(BaseModel):
    """Inference request schema"""
    model_id: str = Field(..., description="Model ID for inference")
    input_data: Union[str, List[float], Dict[str, Any]] = Field(..., description="Input data")
    parameters: Dict[str, Any] = Field(default={}, description="Inference parameters")
    return_probabilities: bool = Field(default=False, description="Return prediction probabilities")

class InferenceResponse(BaseModel):
    """Inference response schema"""
    model_id: str = Field(..., description="Model ID used")
    prediction: Any = Field(..., description="Model prediction")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")

# ==================== DOCUMENT MODELS ====================

class DocumentBase(BaseModel):
    """Base document schema"""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    document_type: str = Field(default="general", description="Document type")
    language: str = Field(default="en", description="Document language")

class DocumentCreate(DocumentBase):
    """Document creation schema"""
    file_path: Optional[str] = Field(None, description="File path if uploaded")

class DocumentResponse(DocumentBase):
    """Document response schema"""
    id: str = Field(..., description="Document ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    summary: Optional[str] = Field(None, description="Document summary")
    keywords: List[str] = Field(default=[], description="Extracted keywords")
    entities: List[Dict[str, Any]] = Field(default=[], description="Named entities")
    sentiment: Optional[Dict[str, Any]] = Field(None, description="Sentiment analysis")

# ==================== CREWAI MODELS ====================

class AgentBase(BaseModel):
    """Base agent schema for CrewAI"""
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role")
    goal: str = Field(..., description="Agent goal")
    backstory: str = Field(..., description="Agent backstory")
    tools: List[str] = Field(default=[], description="Available tools")

class CrewBase(BaseModel):
    """Base crew schema for CrewAI"""
    name: str = Field(..., description="Crew name")
    description: str = Field(..., description="Crew description")
    agents: List[AgentBase] = Field(..., description="Crew agents")
    tasks: List[str] = Field(default=[], description="Crew tasks")

class CrewExecutionRequest(BaseModel):
    """Crew execution request schema"""
    crew_id: str = Field(..., description="Crew ID")
    task_input: str = Field(..., description="Task input")
    max_iterations: int = Field(default=5, description="Maximum iterations")

class CrewExecutionResponse(BaseModel):
    """Crew execution response schema"""
    crew_id: str = Field(..., description="Crew ID")
    task_input: str = Field(..., description="Task input")
    result: Any = Field(..., description="Execution result")
    execution_time: float = Field(..., description="Execution time in seconds")
    iterations: int = Field(..., description="Number of iterations")
    agent_outputs: List[Dict[str, Any]] = Field(default=[], description="Individual agent outputs")

# ==================== LLAMAINDEX MODELS ====================

class IndexBase(BaseModel):
    """Base index schema for LlamaIndex"""
    name: str = Field(..., description="Index name")
    description: str = Field(..., description="Index description")
    index_type: str = Field(default="vector", description="Index type")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")

class IndexCreate(IndexBase):
    """Index creation schema"""
    documents: List[str] = Field(..., description="Documents to index")
    chunk_size: int = Field(default=1024, description="Chunk size for indexing")

class IndexResponse(IndexBase):
    """Index response schema"""
    id: str = Field(..., description="Index ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    document_count: int = Field(..., description="Number of documents indexed")
    size_mb: float = Field(..., description="Index size in MB")

class QueryRequest(BaseModel):
    """Query request schema for LlamaIndex"""
    index_id: str = Field(..., description="Index ID to query")
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")

class QueryResponse(BaseModel):
    """Query response schema for LlamaIndex"""
    index_id: str = Field(..., description="Index ID queried")
    query: str = Field(..., description="Query text")
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    execution_time: float = Field(..., description="Query execution time")

# ==================== ANALYTICS MODELS ====================

class AnalyticsRequest(BaseModel):
    """Analytics request schema"""
    metric_type: str = Field(..., description="Type of analytics metric")
    time_range: str = Field(default="7d", description="Time range for analytics")
    filters: Dict[str, Any] = Field(default={}, description="Analytics filters")

class AnalyticsResponse(BaseModel):
    """Analytics response schema"""
    metric_type: str = Field(..., description="Analytics metric type")
    data: List[Dict[str, Any]] = Field(..., description="Analytics data")
    summary: Dict[str, Any] = Field(default={}, description="Analytics summary")
    generated_at: datetime = Field(..., description="Analytics generation time")

# ==================== WEBHOOK MODELS ====================

class WebhookBase(BaseModel):
    """Base webhook schema"""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to listen for")
    secret: Optional[str] = Field(None, description="Webhook secret")

class WebhookCreate(WebhookBase):
    """Webhook creation schema"""
    pass

class WebhookResponse(WebhookBase):
    """Webhook response schema"""
    id: str = Field(..., description="Webhook ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_active: bool = Field(default=True, description="Webhook active status")

# ==================== ERROR MODELS ====================

class ErrorResponse(BaseModel):
    """Error response schema"""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# ==================== HEALTH CHECK MODELS ====================

class HealthCheck(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(..., description="Service version")
    services: Dict[str, str] = Field(default={}, description="Dependent services status")

# ==================== VALIDATORS ====================

class ModelCreateValidator(ModelCreate):
    """Model creation with validation"""
    
    @validator('name')
    def validate_name(cls, v):
        if len(v) < 3:
            raise ValueError('Model name must be at least 3 characters long')
        return v
    
    @validator('hyperparameters')
    def validate_hyperparameters(cls, v):
        if not isinstance(v, dict):
            raise ValueError('Hyperparameters must be a dictionary')
        return v
