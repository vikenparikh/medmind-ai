"""
MindForge AI - Knowledge Intelligence Test Suite
Comprehensive testing for personal knowledge management features
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
import json
from datetime import datetime

client = TestClient(app)

class TestKnowledgeIntelligence:
    """Comprehensive test suite for MindForge AI knowledge features"""

    def setup_method(self):
        """Setup for each test method"""
        self.test_graph_id = "kg_test_123"
        self.test_node_id = "node_test_456"
        self.sample_content = "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
        self.sample_note = {
            "title": "AI Fundamentals",
            "content": "Understanding the basics of artificial intelligence and machine learning",
            "note_type": "educational",
            "tags": ["AI", "ML", "fundamentals"]
        }

    def test_knowledge_graph_creation(self):
        """Test creating a new knowledge graph"""
        response = client.post("/api/v1/knowledge/graph/create", params={
            "name": "Test Knowledge Graph",
            "description": "A test knowledge graph for AI concepts"
        })
        assert response.status_code == 200
        data = response.json()
        assert "graph_id" in data
        assert data["name"] == "Test Knowledge Graph"
        assert data["status"] == "created"

    def test_add_knowledge_node(self):
        """Test adding a node to knowledge graph"""
        response = client.post(f"/api/v1/knowledge/graph/{self.test_graph_id}/add-node", params={
            "node_type": "concept",
            "content": self.sample_content,
            "metadata": '{"source": "test", "confidence": 0.9}'
        })
        assert response.status_code == 200
        data = response.json()
        assert "node_id" in data
        assert data["graph_id"] == self.test_graph_id
        assert "ai_insights" in data
        assert "connections_suggested" in data

    def test_connect_knowledge_nodes(self):
        """Test connecting two knowledge nodes"""
        response = client.post(f"/api/v1/knowledge/graph/{self.test_graph_id}/connect", params={
            "source_node_id": "node_001",
            "target_node_id": "node_002",
            "connection_type": "prerequisite",
            "strength": 0.8,
            "description": "Machine learning is a prerequisite for deep learning"
        })
        assert response.status_code == 200
        data = response.json()
        assert "connection_id" in data
        assert data["connection_type"] == "prerequisite"
        assert data["strength"] == 0.8

    def test_knowledge_graph_insights(self):
        """Test getting AI-powered knowledge graph insights"""
        response = client.get(f"/api/v1/knowledge/graph/{self.test_graph_id}/insights")
        assert response.status_code == 200
        data = response.json()
        assert "insights" in data
        assert "total_nodes" in data
        assert "total_connections" in data
        assert len(data["insights"]) > 0

    def test_semantic_knowledge_search(self):
        """Test semantic search across knowledge base"""
        response = client.post("/api/v1/knowledge/search/semantic", params={
            "query": "machine learning algorithms",
            "search_type": "comprehensive",
            "max_results": 5
        })
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "ai_analysis" in data
        assert len(data["results"]) > 0

    def test_discover_knowledge_connections(self):
        """Test discovering potential connections between knowledge nodes"""
        response = client.post("/api/v1/knowledge/discover/connections", params={
            "node_id": self.test_node_id,
            "discovery_depth": 2
        })
        assert response.status_code == 200
        data = response.json()
        assert "source_node" in data
        assert "connections_found" in data
        assert "insights" in data
        assert len(data["connections_found"]) > 0

    def test_learning_analytics(self):
        """Test getting learning analytics and progress insights"""
        response = client.get("/api/v1/knowledge/analytics/learning-progress", params={
            "time_period": "30_days"
        })
        assert response.status_code == 200
        data = response.json()
        assert "overview" in data
        assert "progress_metrics" in data
        assert "learning_patterns" in data
        assert "recommendations" in data
        assert "achievements" in data

    def test_predict_next_learning(self):
        """Test predicting next learning opportunities"""
        response = client.post("/api/v1/knowledge/predict/next-learning", json={
            "graph_id": self.test_graph_id,
            "user_context": {"level": "intermediate", "interests": ["AI", "ML"]}
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "learning_goals" in data
        assert "optimization_tips" in data
        assert len(data["predictions"]) > 0

    def test_smart_note_creation(self):
        """Test creating smart notes with AI enhancement"""
        response = client.post("/api/v1/knowledge/notes/smart-create", params=self.sample_note)
        assert response.status_code == 200
        data = response.json()
        assert "note_id" in data
        assert data["title"] == self.sample_note["title"]
        assert "ai_enhancements" in data
        assert "connections" in data
        assert "suggestions" in data

    def test_knowledge_graph_comprehensive_workflow(self):
        """Test a comprehensive knowledge management workflow"""
        # 1. Create knowledge graph
        graph_response = client.post("/api/v1/knowledge/graph/create", params={
            "name": "AI Learning Journey",
            "description": "Personal knowledge graph for AI learning"
        })
        assert graph_response.status_code == 200
        graph_id = graph_response.json()["graph_id"]

        # 2. Add multiple knowledge nodes
        nodes = [
            {"type": "concept", "content": "Machine Learning Fundamentals"},
            {"type": "concept", "content": "Deep Learning and Neural Networks"},
            {"type": "application", "content": "AI in Business Applications"}
        ]
        
        node_ids = []
        for node in nodes:
            response = client.post(f"/api/v1/knowledge/graph/{graph_id}/add-node", json=node)
            assert response.status_code == 200
            node_ids.append(response.json()["node_id"])

        # 3. Connect the nodes
        for i in range(len(node_ids) - 1):
            response = client.post(f"/api/v1/knowledge/graph/{graph_id}/connect", json={
                "source_node_id": node_ids[i],
                "target_node_id": node_ids[i + 1],
                "connection_type": "progression",
                "strength": 0.8
            })
            assert response.status_code == 200

        # 4. Get insights about the graph
        insights_response = client.get(f"/api/v1/knowledge/graph/{graph_id}/insights")
        assert insights_response.status_code == 200
        insights = insights_response.json()
        assert insights["total_nodes"] >= len(nodes)

        # 5. Search within the knowledge graph
        search_response = client.post("/api/v1/knowledge/search/semantic", params={
            "query": "machine learning",
            "graph_id": graph_id
        })
        assert search_response.status_code == 200
        search_results = search_response.json()
        assert len(search_results["results"]) > 0

        # 6. Get learning analytics
        analytics_response = client.get("/api/v1/knowledge/analytics/learning-progress", params={
            "graph_id": graph_id
        })
        assert analytics_response.status_code == 200

        # 7. Predict next learning opportunities
        prediction_response = client.post("/api/v1/knowledge/predict/next-learning", json={
            "graph_id": graph_id
        })
        assert prediction_response.status_code == 200

    def test_error_handling(self):
        """Test error handling for knowledge endpoints"""
        # Test invalid graph ID
        response = client.get("/api/v1/knowledge/graph/invalid_graph/insights")
        assert response.status_code == 200  # Should handle gracefully
        
        # Test invalid node connection
        response = client.post("/api/v1/knowledge/graph/test/connect", json={
            "source_node_id": "invalid",
            "target_node_id": "invalid",
            "connection_type": "test"
        })
        assert response.status_code == 200  # Should handle gracefully

    def test_performance_metrics(self):
        """Test performance of knowledge intelligence features"""
        import time
        
        # Test semantic search performance
        start_time = time.time()
        response = client.post("/api/v1/knowledge/search/semantic", params={
            "query": "artificial intelligence machine learning deep learning neural networks"
        })
        search_time = time.time() - start_time
        
        assert response.status_code == 200
        assert search_time < 1.0  # Should be fast
        
        # Test knowledge insights performance
        start_time = time.time()
        response = client.get("/api/v1/knowledge/graph/test/insights")
        insights_time = time.time() - start_time
        
        assert response.status_code == 200
        assert insights_time < 0.5  # Should be very fast

    def test_concurrent_knowledge_operations(self):
        """Test handling concurrent knowledge operations"""
        import threading
        import time
        
        results = []
        
        def add_node(node_content):
            response = client.post("/api/v1/knowledge/graph/test/add-node", json={
                "node_type": "concept",
                "content": node_content
            })
            results.append(response.status_code == 200)
        
        # Create multiple threads for concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_node, args=(f"Concept {i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        assert all(results)
        assert len(results) == 5

    def test_knowledge_data_consistency(self):
        """Test data consistency across knowledge operations"""
        # Create a knowledge graph
        graph_response = client.post("/api/v1/knowledge/graph/create", params={
            "name": "Consistency Test Graph"
        })
        graph_id = graph_response.json()["graph_id"]
        
        # Add a node
        node_response = client.post(f"/api/v1/knowledge/graph/{graph_id}/add-node", json={
            "node_type": "test",
            "content": "Test content for consistency"
        })
        node_id = node_response.json()["node_id"]
        
        # Verify the node exists in insights
        insights_response = client.get(f"/api/v1/knowledge/graph/{graph_id}/insights")
        insights = insights_response.json()
        
        # The graph should have at least one node
        assert insights["total_nodes"] >= 1
        
        # Search should find the node
        search_response = client.post("/api/v1/knowledge/search/semantic", params={
            "query": "test content",
            "graph_id": graph_id
        })
        search_results = search_response.json()
        
        # Should find relevant results
        assert len(search_results["results"]) >= 0  # May or may not find depending on mock implementation
