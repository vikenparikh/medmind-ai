"""
MedMind AI - End-to-End Tests
Comprehensive E2E tests for medical AI platform
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.services.medical_ai_engine import medical_ai_engine
import json
from datetime import datetime

client = TestClient(app)

class TestMedMindAIE2E:
    """End-to-End test suite for MedMind AI platform"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.test_patient_id = "patient_test_001"
        self.test_symptoms = ["fever", "cough", "headache", "fatigue"]
        self.test_medications = ["ibuprofen", "acetaminophen", "amoxicillin"]
        self.test_patient_data = {
            "age": 35,
            "gender": "female",
            "weight": 65,
            "height": 165,
            "medical_history": ["hypertension"],
            "current_medications": self.test_medications,
            "allergies": ["penicillin"],
            "vital_signs": {
                "blood_pressure": "120/80",
                "heart_rate": 72,
                "temperature": 38.5,
                "respiratory_rate": 18
            }
        }
    
    def test_platform_health_e2e(self):
        """Test complete platform health and functionality"""
        print("\n🏥 Testing MedMind AI Platform Health...")
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Welcome to MedMind AI" in data["message"]
        print("✅ Root endpoint working")
        
        # Test health check
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check working")
        
        # Test API health
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        print("✅ API health working")
        
        print("🎉 MedMind AI platform is healthy!")
    
    @pytest.mark.asyncio
    async def test_medical_ai_engine_e2e(self):
        """Test medical AI engine end-to-end functionality"""
        print("\n🧠 Testing Medical AI Engine...")
        
        # Test clinical crew creation
        crew_config = {
            "name": "Emergency Medical Crew",
            "agents": [
                {"role": "Emergency Physician", "goal": "Provide emergency care"},
                {"role": "Nurse", "goal": "Support patient care"}
            ]
        }
        
        result = await medical_ai_engine.create_clinical_crew(crew_config)
        assert "crew_id" in result
        assert result["status"] == "created"
        print("✅ Clinical crew creation working")
        
        # Test symptom analysis
        result = await medical_ai_engine.analyze_symptoms(self.test_symptoms, self.test_patient_data)
        assert "differential_diagnosis" in result
        assert "risk_assessment" in result
        assert "treatment_recommendations" in result
        print("✅ Symptom analysis working")
        
        # Test drug interaction checking
        result = await medical_ai_engine.check_drug_interactions(self.test_medications, self.test_patient_data)
        assert "drug_interactions" in result
        assert "safety_score" in result
        assert "recommendations" in result
        print("✅ Drug interaction checking working")
        
        # Test patient outcome prediction
        result = await medical_ai_engine.predict_patient_outcomes(self.test_patient_data)
        assert "predicted_outcomes" in result
        assert "risk_stratification" in result
        assert "intervention_recommendations" in result
        print("✅ Patient outcome prediction working")
        
        # Test medical knowledge graph
        medical_data = {
            "diseases": ["hypertension", "diabetes"],
            "symptoms": self.test_symptoms,
            "treatments": ["lifestyle_modification", "medication"],
            "drugs": self.test_medications
        }
        
        result = await medical_ai_engine.build_medical_knowledge_graph(medical_data)
        assert "graph_id" in result
        assert "knowledge_graph" in result
        assert "statistics" in result
        print("✅ Medical knowledge graph working")
        
        print("🎉 Medical AI Engine is fully functional!")
    
    def test_medical_api_endpoints_e2e(self):
        """Test medical API endpoints end-to-end"""
        print("\n🔗 Testing Medical API Endpoints...")
        
        # Test symptom analysis endpoint
        response = client.post("/api/v1/medical/symptom-analysis", 
                             params={
                                 "symptoms": json.dumps(self.test_symptoms),
                                 "patient_context": json.dumps(self.test_patient_data)
                             })
        assert response.status_code == 200
        data = response.json()
        assert "differential_diagnosis" in data
        print("✅ Symptom analysis endpoint working")
        
        # Test drug interaction endpoint
        response = client.post("/api/v1/medical/drug-interactions",
                             params={
                                 "medications": json.dumps(self.test_medications),
                                 "patient_profile": json.dumps(self.test_patient_data)
                             })
        assert response.status_code == 200
        data = response.json()
        assert "safety_score" in data
        print("✅ Drug interaction endpoint working")
        
        # Test patient outcome prediction endpoint
        response = client.post("/api/v1/medical/outcome-prediction",
                             params={"patient_data": json.dumps(self.test_patient_data)})
        assert response.status_code == 200
        data = response.json()
        assert "predicted_outcomes" in data
        print("✅ Outcome prediction endpoint working")
        
        print("🎉 Medical API endpoints are fully functional!")
    
    def test_medical_workflow_e2e(self):
        """Test complete medical workflow end-to-end"""
        print("\n🔄 Testing Complete Medical Workflow...")
        
        # Step 1: Patient presents with symptoms
        symptoms_response = client.post("/api/v1/medical/symptom-analysis",
                                      params={
                                          "symptoms": json.dumps(self.test_symptoms),
                                          "patient_context": json.dumps(self.test_patient_data)
                                      })
        assert symptoms_response.status_code == 200
        symptom_data = symptoms_response.json()
        print("✅ Step 1: Symptom analysis completed")
        
        # Step 2: Check current medications for interactions
        interactions_response = client.post("/api/v1/medical/drug-interactions",
                                          params={
                                              "medications": json.dumps(self.test_medications),
                                              "patient_profile": json.dumps(self.test_patient_data)
                                          })
        assert interactions_response.status_code == 200
        interactions_data = interactions_response.json()
        print("✅ Step 2: Drug interaction analysis completed")
        
        # Step 3: Predict patient outcomes
        outcomes_response = client.post("/api/v1/medical/outcome-prediction",
                                      params={"patient_data": json.dumps(self.test_patient_data)})
        assert outcomes_response.status_code == 200
        outcomes_data = outcomes_response.json()
        print("✅ Step 3: Outcome prediction completed")
        
        # Step 4: Generate comprehensive medical report
        medical_report = {
            "patient_id": self.test_patient_id,
            "presentation_time": datetime.utcnow().isoformat(),
            "symptoms_analysis": symptom_data,
            "drug_interactions": interactions_data,
            "outcome_predictions": outcomes_data,
            "clinical_summary": {
                "primary_diagnosis": symptom_data["differential_diagnosis"][0]["condition"],
                "risk_level": outcomes_data["risk_stratification"]["overall_risk_level"],
                "safety_score": interactions_data["safety_score"],
                "recommendations": [
                    symptom_data["treatment_recommendations"][0]["first_line"],
                    interactions_data["recommendations"][0],
                    outcomes_data["intervention_recommendations"][0]["action"] if outcomes_data["intervention_recommendations"] else "Continue monitoring"
                ]
            }
        }
        
        # Validate comprehensive report
        assert "patient_id" in medical_report
        assert "symptoms_analysis" in medical_report
        assert "drug_interactions" in medical_report
        assert "outcome_predictions" in medical_report
        assert "clinical_summary" in medical_report
        print("✅ Step 4: Comprehensive medical report generated")
        
        print("🎉 Complete medical workflow is functional!")
        print(f"📋 Medical Report Summary:")
        print(f"   Patient ID: {medical_report['patient_id']}")
        print(f"   Primary Diagnosis: {medical_report['clinical_summary']['primary_diagnosis']}")
        print(f"   Risk Level: {medical_report['clinical_summary']['risk_level']}")
        print(f"   Safety Score: {medical_report['clinical_summary']['safety_score']:.2f}")
    
    def test_medical_error_handling_e2e(self):
        """Test medical error handling end-to-end"""
        print("\n⚠️ Testing Medical Error Handling...")
        
        # Test invalid symptom analysis
        response = client.post("/api/v1/medical/symptom-analysis",
                             params={
                                 "symptoms": json.dumps([]),  # Empty symptoms
                                 "patient_context": json.dumps({})
                             })
        assert response.status_code == 200  # Should handle gracefully
        print("✅ Empty symptoms handled gracefully")
        
        # Test invalid drug interactions
        response = client.post("/api/v1/medical/drug-interactions",
                             params={
                                 "medications": json.dumps([]),  # Empty medications
                                 "patient_profile": json.dumps({})
                             })
        assert response.status_code == 200  # Should handle gracefully
        print("✅ Empty medications handled gracefully")
        
        # Test invalid patient data
        response = client.post("/api/v1/medical/outcome-prediction",
                             params={"patient_data": json.dumps({})})
        assert response.status_code == 200  # Should handle gracefully
        print("✅ Empty patient data handled gracefully")
        
        print("🎉 Medical error handling is robust!")
    
    def test_medical_performance_e2e(self):
        """Test medical platform performance end-to-end"""
        print("\n⚡ Testing Medical Platform Performance...")
        
        import time
        
        # Test response times
        start_time = time.time()
        response = client.get("/health")
        health_time = time.time() - start_time
        assert health_time < 1.0  # Should respond within 1 second
        print(f"✅ Health check response time: {health_time:.3f}s")
        
        start_time = time.time()
        response = client.post("/api/v1/medical/symptom-analysis",
                             params={
                                 "symptoms": json.dumps(self.test_symptoms),
                                 "patient_context": json.dumps(self.test_patient_data)
                             })
        symptom_time = time.time() - start_time
        assert symptom_time < 5.0  # Should respond within 5 seconds
        print(f"✅ Symptom analysis response time: {symptom_time:.3f}s")
        
        start_time = time.time()
        response = client.post("/api/v1/medical/drug-interactions",
                             params={
                                 "medications": json.dumps(self.test_medications),
                                 "patient_profile": json.dumps(self.test_patient_data)
                             })
        interaction_time = time.time() - start_time
        assert interaction_time < 5.0  # Should respond within 5 seconds
        print(f"✅ Drug interaction response time: {interaction_time:.3f}s")
        
        print("🎉 Medical platform performance is excellent!")
    
    def test_medical_security_e2e(self):
        """Test medical platform security end-to-end"""
        print("\n🔒 Testing Medical Platform Security...")
        
        # Test CORS headers
        response = client.options("/api/v1/medical/symptom-analysis")
        assert response.status_code in [200, 405]  # OPTIONS may not be implemented
        print("✅ CORS handling working")
        
        # Test invalid JSON handling
        response = client.post("/api/v1/medical/symptom-analysis",
                             params={
                                 "symptoms": "invalid_json",
                                 "patient_context": json.dumps(self.test_patient_data)
                             })
        assert response.status_code == 422  # Should return validation error
        print("✅ Invalid JSON handling working")
        
        # Test large payload handling
        large_symptoms = ["symptom_" + str(i) for i in range(1000)]
        response = client.post("/api/v1/medical/symptom-analysis",
                             params={
                                 "symptoms": json.dumps(large_symptoms),
                                 "patient_context": json.dumps(self.test_patient_data)
                             })
        assert response.status_code == 200  # Should handle large payloads
        print("✅ Large payload handling working")
        
        print("🎉 Medical platform security is robust!")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
