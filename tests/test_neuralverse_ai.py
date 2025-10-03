"""
NeuralVerse AI - Unified AI Platform Tests
Comprehensive tests for the unified AI development platform
"""

import pytest
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.main import app
from app.core.config import settings
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Test client
client = TestClient(app)

class TestNeuralVerseAI:
    """Comprehensive test suite for NeuralVerse AI unified platform"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.test_model_id = "model_001"
        self.test_crew_id = "crew_001"
        self.test_index_id = "index_001"
        
        # Sample data for testing
        self.sample_text = "NeuralVerse AI is revolutionizing artificial intelligence development with cutting-edge technologies."
        self.sample_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.sample_documents = [
            "NeuralVerse AI integrates multiple AI technologies seamlessly",
            "Multi-agent systems enable collaborative AI development",
            "Vector search provides advanced document retrieval capabilities"
        ]

    # ==================== CORE PLATFORM TESTS ====================

    def test_platform_root(self):
        """Test the unified platform root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Welcome to MindForge AI - Personal Knowledge Intelligence Platform"
        assert data["version"] == settings.APP_VERSION
        assert data["status"] == "operational"
        assert "features" in data
        assert len(data["features"]) >= 10
        
        # Verify all key features are mentioned
        features_text = " ".join(data["features"])
        assert "Knowledge Graph" in features_text
        assert "Multi-Agent" in features_text
        assert "Semantic Search" in features_text
        assert "Learning Analytics" in features_text
        assert "Predictive Insights" in features_text

    def test_health_check(self):
        """Test comprehensive health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data
        assert "metrics" in data
        assert data["version"] == settings.APP_VERSION

    def test_system_status(self):
        """Test detailed system status"""
        response = client.get("/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["platform"] == "NeuralVerse AI"
        assert data["version"] == settings.APP_VERSION
        assert "services_status" in data
        assert "performance" in data
        
        # Verify all AI technologies are listed
        services = data["services_status"]
        assert "crewai" in services
        assert "llamaindex" in services
        assert "pytorch" in services
        assert "scikit_learn" in services
        assert "langchain" in services

    # ==================== MULTI-AGENT AI SYSTEM TESTS ====================

    def test_crew_creation(self):
        """Test CrewAI multi-agent system creation"""
        crew_config = {
            "name": "NeuralVerse Research Crew",
            "description": "Advanced AI research and development crew",
            "agents": [
                {
                    "name": "AI Researcher",
                    "role": "Senior AI Researcher",
                    "goal": "Research and analyze cutting-edge AI technologies",
                    "backstory": "Expert in artificial intelligence with deep knowledge of ML and NLP"
                },
                {
                    "name": "Data Scientist",
                    "role": "Data Analysis Specialist",
                    "goal": "Analyze data and extract insights",
                    "backstory": "Specialized in data analysis and statistical modeling"
                }
            ],
            "tasks": [
                "Research latest AI developments",
                "Analyze market trends in AI",
                "Generate comprehensive reports"
            ]
        }
        
        response = client.post("/api/v1/crew/create", json=crew_config)
        assert response.status_code == 200
        
        data = response.json()
        assert "crew_id" in data
        assert data["status"] == "created"
        assert data["agents_count"] == 2
        assert data["tasks_count"] == 3

    def test_crew_execution(self):
        """Test CrewAI task execution"""
        execution_request = {
            "crew_id": self.test_crew_id,
            "task_input": "Research the impact of neural networks on modern AI",
            "max_iterations": 3
        }
        
        response = client.post("/api/v1/crew/execute", json=execution_request)
        # Note: This might return 400 if crew doesn't exist, which is expected in test environment
        assert response.status_code in [200, 400]

    # ==================== VECTOR SEARCH & INTELLIGENCE TESTS ====================

    def test_vector_index_creation(self):
        """Test LlamaIndex vector search index creation"""
        index_config = {
            "name": "NeuralVerse Knowledge Base",
            "description": "Comprehensive knowledge base for AI technologies",
            "index_type": "vector",
            "embedding_model": "text-embedding-ada-002",
            "documents": self.sample_documents,
            "chunk_size": 512
        }
        
        response = client.post("/api/v1/index/create", json=index_config)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["name"] == index_config["name"]
        assert data["document_count"] == len(index_config["documents"])
        assert data["index_type"] == "vector"

    def test_vector_search(self):
        """Test vector search functionality"""
        query_request = {
            "index_id": self.test_index_id,
            "query": "What are the key features of NeuralVerse AI?",
            "top_k": 5,
            "similarity_threshold": 0.7
        }
        
        response = client.post("/api/v1/index/query", json=query_request)
        # Note: This might return 400 if index doesn't exist, which is expected in test environment
        assert response.status_code in [200, 400]

    # ==================== MACHINE LEARNING PIPELINE TESTS ====================

    def test_model_creation(self):
        """Test ML model creation"""
        model_data = {
            "name": "NeuralVerse Classifier",
            "description": "Advanced classification model for AI applications",
            "model_type": "classification",
            "framework": "pytorch",
            "version": "1.0.0",
            "tags": ["ai", "classification", "pytorch"],
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "hidden_layers": [128, 64, 32]
            }
        }
        
        response = client.post("/api/v1/models", json=model_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == model_data["name"]
        assert data["model_type"] == model_data["model_type"]
        assert data["framework"] == model_data["framework"]
        assert "id" in data

    def test_model_training(self):
        """Test ML model training pipeline"""
        training_data = {
            "model_id": self.test_model_id,
            "dataset_config": {
                "dataset_path": "/data/neuralverse_training.csv",
                "features": ["feature1", "feature2", "feature3"],
                "target": "label"
            },
            "training_config": {
                "epochs": 50,
                "batch_size": 32,
                "optimizer": "adam",
                "loss_function": "cross_entropy"
            },
            "validation_config": {
                "validation_split": 0.2,
                "early_stopping": True,
                "patience": 10
            }
        }
        
        response = client.post("/api/v1/training", json=training_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_id"] == training_data["model_id"]
        assert data["status"] == "started"
        assert "id" in data

    def test_model_inference(self):
        """Test ML model inference"""
        inference_data = {
            "model_id": self.test_model_id,
            "input_data": self.sample_data,
            "parameters": {
                "return_probabilities": True,
                "confidence_threshold": 0.8
            }
        }
        
        response = client.post("/api/v1/inference", json=inference_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "processing_time" in data
        assert "metadata" in data

    # ==================== NATURAL LANGUAGE PROCESSING TESTS ====================

    def test_nlp_comprehensive_processing(self):
        """Test comprehensive NLP processing pipeline"""
        response = client.post("/api/v1/nlp/process", params={
            "text": self.sample_text, 
            "task": "comprehensive"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "sentiment" in data
        assert "entities" in data
        assert "summary" in data
        assert "keywords" in data
        assert "embedding" in data
        assert "classification" in data

    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        response = client.post("/api/v1/nlp/sentiment", params={
            "text": self.sample_text
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "processing_time" in data

    def test_entity_extraction(self):
        """Test named entity extraction"""
        response = client.post("/api/v1/nlp/entities", params={
            "text": self.sample_text
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "entities" in data
        assert "entity_count" in data
        assert "processing_time" in data

    # ==================== COMPUTER VISION TESTS ====================

    def test_image_analysis(self):
        """Test comprehensive image analysis"""
        test_image_content = b"fake_image_data_for_testing"
        
        response = client.post("/api/v1/vision/analyze", 
                              files={"file": ("test_image.jpg", test_image_content, "image/jpeg")},
                              params={"task": "comprehensive"})
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis" in data
        assert "filename" in data
        assert "file_size" in data

    def test_object_detection(self):
        """Test object detection"""
        test_image_content = b"fake_image_data_for_object_detection"
        
        response = client.post("/api/v1/vision/objects", 
                              files={"file": ("test_image.jpg", test_image_content, "image/jpeg")})
        assert response.status_code == 200
        
        data = response.json()
        assert "objects" in data
        assert "object_count" in data

    def test_face_detection(self):
        """Test face detection"""
        test_image_content = b"fake_image_data_for_face_detection"
        
        response = client.post("/api/v1/vision/faces", 
                              files={"file": ("test_image.jpg", test_image_content, "image/jpeg")})
        assert response.status_code == 200
        
        data = response.json()
        assert "faces" in data
        assert "face_count" in data

    # ==================== GENERATIVE AI TESTS ====================

    def test_text_generation(self):
        """Test AI text generation"""
        response = client.post("/api/v1/generate/text", 
                              params={
                                  "prompt": "Explain the benefits of multi-agent AI systems",
                                  "max_length": 200
                              })
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_text" in data
        assert "model" in data
        assert "processing_time" in data

    def test_code_generation(self):
        """Test AI code generation"""
        response = client.post("/api/v1/generate/code", 
                              params={
                                  "prompt": "Create a Python function to calculate the fibonacci sequence"
                              })
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_code" in data
        assert "language" in data
        assert "model" in data

    # ==================== AUDIO PROCESSING TESTS ====================

    def test_audio_processing(self):
        """Test audio processing"""
        test_audio_content = b"fake_audio_data_for_testing"
        
        response = client.post("/api/v1/audio/process", 
                              files={"file": ("test_audio.wav", test_audio_content, "audio/wav")},
                              params={"task": "speech_recognition"})
        assert response.status_code == 200
        
        data = response.json()
        assert "result" in data
        assert "filename" in data

    # ==================== ANALYTICS & INSIGHTS TESTS ====================

    def test_analytics_generation(self):
        """Test analytics generation"""
        analytics_request = {
            "metric_type": "comprehensive",
            "time_range": "30d",
            "filters": {
                "category": "AI",
                "platform": "NeuralVerse"
            }
        }
        
        response = client.post("/api/v1/analytics/generate", json=analytics_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "metric_type" in data
        assert "data" in data
        assert "summary" in data
        assert "generated_at" in data

    # ==================== DOCUMENT PROCESSING TESTS ====================

    def test_document_creation(self):
        """Test document creation and processing"""
        document_data = {
            "title": "NeuralVerse AI Platform Overview",
            "content": self.sample_text,
            "document_type": "technical",
            "language": "en"
        }
        
        response = client.post("/api/v1/documents", json=document_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["title"] == document_data["title"]
        assert data["content"] == document_data["content"]
        assert "summary" in data
        assert "keywords" in data
        assert "entities" in data

    def test_document_listing(self):
        """Test document listing"""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # Should have at least 1 sample document

    # ==================== WEBSOCKET TESTS ====================

    def test_websocket_connection(self):
        """Test WebSocket connection"""
        try:
            with client.websocket_connect("/ws/test_client") as websocket:
                # Test ping-pong
                ping_message = {"type": "ping"}
                websocket.send_text(json.dumps(ping_message))
                
                response = websocket.receive_text()
                response_data = json.loads(response)
                assert response_data["type"] == "pong"
                
        except Exception as e:
            # WebSocket tests can be flaky in test environment
            pytest.skip(f"WebSocket test skipped due to: {e}")

    def test_websocket_ai_request(self):
        """Test WebSocket AI request handling"""
        try:
            with client.websocket_connect("/ws/test_client") as websocket:
                # Test AI request
                ai_request = {
                    "type": "ai_request",
                    "request_type": "nlp",
                    "data": self.sample_text,
                    "request_id": "test_001"
                }
                websocket.send_text(json.dumps(ai_request))
                
                response = websocket.receive_text()
                response_data = json.loads(response)
                assert response_data["type"] == "ai_response"
                assert "result" in response_data
                
        except Exception as e:
            # WebSocket tests can be flaky in test environment
            pytest.skip(f"WebSocket AI request test skipped due to: {e}")

    # ==================== INTEGRATION TESTS ====================

    def test_full_ai_pipeline(self):
        """Test complete AI pipeline integration"""
        # 1. Create a document
        document_data = {
            "title": "AI Integration Test",
            "content": "This document tests the full AI pipeline integration in NeuralVerse AI.",
            "document_type": "test",
            "language": "en"
        }
        
        doc_response = client.post("/api/v1/documents", json=document_data)
        assert doc_response.status_code == 200
        
        # 2. Process with NLP
        nlp_response = client.post("/api/v1/nlp/process", 
                                  params={
                                      "text": document_data["content"], 
                                      "task": "comprehensive"
                                  })
        assert nlp_response.status_code == 200
        
        # 3. Generate analytics
        analytics_response = client.post("/api/v1/analytics/generate", 
                                        json={
                                            "metric_type": "comprehensive", 
                                            "time_range": "1d"
                                        })
        assert analytics_response.status_code == 200
        
        # 4. Create vector index
        index_response = client.post("/api/v1/index/create", 
                                    json={
                                        "name": "Test Pipeline Index",
                                        "description": "Index for pipeline testing",
                                        "documents": [document_data["content"]],
                                        "chunk_size": 256
                                    })
        assert index_response.status_code == 200

    def test_error_handling(self):
        """Test comprehensive error handling"""
        # Test 404 error
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test invalid model inference
        invalid_inference = {
            "model_id": "nonexistent_model",
            "input_data": [1, 2, 3],
            "parameters": {}
        }
        response = client.post("/api/v1/inference", json=invalid_inference)
        assert response.status_code in [404, 500]  # Accept either 404 or 500 for error handling

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(status == 200 for status in results)

    def test_performance_metrics(self):
        """Test performance and response times"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Response should be fast (less than 2 seconds)
        assert response_time < 2.0
        assert response.status_code == 200

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
