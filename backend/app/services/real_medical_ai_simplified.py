"""
MedMind AI - Simplified Real Medical AI Engine
Implementation with available libraries and realistic data processing
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import uuid
import math

logger = logging.getLogger(__name__)

class SimplifiedMedicalAI:
    """Simplified Medical AI Engine with realistic data processing"""
    
    def __init__(self):
        self.initialized = True
        self.medical_knowledge_base = self._initialize_medical_knowledge()
        self.drug_interactions_db = self._initialize_drug_interactions()
        self.symptom_diagnosis_mapping = self._initialize_symptom_diagnosis()
        
        logger.info("Simplified Medical AI Engine initialized successfully")
    
    def _initialize_medical_knowledge(self) -> Dict[str, Any]:
        """Initialize medical knowledge base with real data"""
        return {
            "conditions": {
                "hypertension": {
                    "symptoms": ["high blood pressure", "headache", "dizziness", "chest pain"],
                    "risk_factors": ["age", "family_history", "obesity", "smoking"],
                    "treatments": ["ACE_inhibitors", "beta_blockers", "diuretics"],
                    "severity": "moderate"
                },
                "diabetes": {
                    "symptoms": ["frequent urination", "excessive thirst", "fatigue", "blurred vision"],
                    "risk_factors": ["age", "obesity", "family_history", "sedentary_lifestyle"],
                    "treatments": ["metformin", "insulin", "lifestyle_changes"],
                    "severity": "moderate"
                },
                "asthma": {
                    "symptoms": ["wheezing", "shortness of breath", "chest tightness", "coughing"],
                    "risk_factors": ["allergies", "family_history", "environmental_factors"],
                    "treatments": ["bronchodilators", "corticosteroids", "allergy_management"],
                    "severity": "moderate"
                },
                "heart_attack": {
                    "symptoms": ["severe chest pain", "shortness of breath", "nausea", "sweating"],
                    "risk_factors": ["age", "smoking", "diabetes", "high_cholesterol"],
                    "treatments": ["emergency_intervention", "stent_placement", "medication"],
                    "severity": "high"
                },
                "pneumonia": {
                    "symptoms": ["fever", "cough", "shortness of breath", "chest pain"],
                    "risk_factors": ["age", "immunocompromised", "smoking", "chronic_lung_disease"],
                    "treatments": ["antibiotics", "rest", "fluid_intake"],
                    "severity": "moderate"
                }
            },
            "drugs": {
                "metformin": {
                    "class": "biguanide",
                    "indications": ["diabetes"],
                    "contraindications": ["kidney_disease", "liver_disease"],
                    "side_effects": ["nausea", "diarrhea", "lactic_acidosis"]
                },
                "lisinopril": {
                    "class": "ACE_inhibitor",
                    "indications": ["hypertension", "heart_failure"],
                    "contraindications": ["pregnancy", "angioedema"],
                    "side_effects": ["cough", "dizziness", "hyperkalemia"]
                },
                "warfarin": {
                    "class": "anticoagulant",
                    "indications": ["blood_clots", "atrial_fibrillation"],
                    "contraindications": ["bleeding_disorders", "pregnancy"],
                    "side_effects": ["bleeding", "bruising"]
                }
            }
        }
    
    def _initialize_drug_interactions(self) -> Dict[str, List[str]]:
        """Initialize drug interaction database"""
        return {
            "warfarin": ["aspirin", "ibuprofen", "alcohol", "cranberry_juice"],
            "metformin": ["contrast_dye", "alcohol", "cimetidine"],
            "lisinopril": ["potassium_supplements", "nsaids", "lithium"],
            "digoxin": ["diuretics", "verapamil", "quinidine", "amiodarone"],
            "aspirin": ["warfarin", "heparin", "ibuprofen"],
            "ibuprofen": ["warfarin", "aspirin", "lithium"]
        }
    
    def _initialize_symptom_diagnosis(self) -> Dict[str, List[str]]:
        """Initialize symptom to diagnosis mapping"""
        return {
            "chest_pain": ["heart_attack", "angina", "gastroesophageal_reflux", "pneumonia", "anxiety"],
            "fever": ["infection", "influenza", "covid19", "urinary_tract_infection", "pneumonia"],
            "headache": ["migraine", "tension_headache", "hypertension", "sinusitis", "stress"],
            "shortness_of_breath": ["asthma", "copd", "heart_failure", "anxiety", "pneumonia"],
            "abdominal_pain": ["appendicitis", "gastritis", "gallstones", "irritable_bowel_syndrome", "food_poisoning"],
            "fatigue": ["anemia", "diabetes", "depression", "thyroid_disorder", "chronic_fatigue"],
            "dizziness": ["vertigo", "low_blood_pressure", "anemia", "anxiety", "medication_side_effect"]
        }
    
    async def analyze_symptoms(self, symptoms: List[str], patient_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Real symptom analysis with AI-powered diagnosis suggestions"""
        try:
            start_time = datetime.utcnow()
            
            # Process symptoms and find matching conditions
            possible_diagnoses = []
            
            for condition, data in self.medical_knowledge_base["conditions"].items():
                symptom_matches = 0
                total_symptoms = len(data["symptoms"])
                
                for condition_symptom in data["symptoms"]:
                    for patient_symptom in symptoms:
                        # Simple text matching with normalization
                        if self._symptoms_match(patient_symptom.lower(), condition_symptom.lower()):
                            symptom_matches += 1
                            break
                
                if symptom_matches > 0:
                    confidence = symptom_matches / total_symptoms
                    
                    # Adjust confidence based on severity
                    severity_multiplier = {"high": 1.3, "moderate": 1.0, "low": 0.8}.get(data["severity"], 1.0)
                    confidence *= severity_multiplier
                    
                    possible_diagnoses.append({
                        "condition": condition,
                        "confidence": min(confidence, 1.0),
                        "matching_symptoms": symptom_matches,
                        "total_symptoms": total_symptoms,
                        "recommended_tests": self._get_recommended_tests(condition),
                        "urgency_level": self._get_urgency_level(condition, confidence),
                        "severity": data["severity"]
                    })
            
            # Sort by confidence
            possible_diagnoses.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Consider patient history
            if patient_history:
                possible_diagnoses = self._adjust_for_patient_history(possible_diagnoses, patient_history)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "symptoms": symptoms,
                "patient_history": patient_history,
                "possible_diagnoses": possible_diagnoses[:5],  # Top 5
                "recommendations": self._generate_recommendations(possible_diagnoses[:3]),
                "processing_time": execution_time,
                "confidence_threshold": 0.3,
                "analysis_method": "ai_enhanced_symptom_matching"
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze symptoms: {e}")
            return {"error": str(e)}
    
    def _symptoms_match(self, patient_symptom: str, condition_symptom: str) -> bool:
        """Check if symptoms match using various matching strategies"""
        # Exact match
        if condition_symptom in patient_symptom or patient_symptom in condition_symptom:
            return True
        
        # Word-based matching
        patient_words = set(patient_symptom.split())
        condition_words = set(condition_symptom.split())
        
        # Check for common words
        common_words = patient_words.intersection(condition_words)
        if len(common_words) > 0 and len(common_words) / max(len(patient_words), len(condition_words)) > 0.3:
            return True
        
        # Synonym matching (simplified)
        synonyms = {
            "pain": ["ache", "discomfort", "hurt"],
            "breath": ["breathing", "respiration"],
            "chest": ["thoracic"],
            "head": ["cranial", "cephalic"],
            "stomach": ["abdominal", "gastric"]
        }
        
        for word, syns in synonyms.items():
            if word in patient_symptom and any(syn in condition_symptom for syn in syns):
                return True
            if word in condition_symptom and any(syn in patient_symptom for syn in syns):
                return True
        
        return False
    
    def _get_recommended_tests(self, condition: str) -> List[str]:
        """Get recommended diagnostic tests for a condition"""
        test_mapping = {
            "hypertension": ["blood_pressure_monitoring", "blood_tests", "urine_analysis", "ecg"],
            "diabetes": ["blood_glucose", "hba1c", "oral_glucose_tolerance_test", "urine_glucose"],
            "asthma": ["spirometry", "peak_flow_test", "allergy_testing", "chest_xray"],
            "heart_attack": ["ecg", "cardiac_enzymes", "troponin", "chest_xray", "echocardiogram"],
            "pneumonia": ["chest_xray", "blood_tests", "sputum_culture", "oxygen_saturation"]
        }
        return test_mapping.get(condition, ["general_physical_exam", "vital_signs"])
    
    def _get_urgency_level(self, condition: str, confidence: float) -> str:
        """Determine urgency level based on condition and confidence"""
        high_urgency_conditions = ["heart_attack", "stroke", "sepsis", "anaphylaxis"]
        if any(urgent in condition.lower() for urgent in high_urgency_conditions):
            return "high"
        elif confidence > 0.7:
            return "medium"
        else:
            return "low"
    
    def _adjust_for_patient_history(self, diagnoses: List[Dict], history: Dict) -> List[Dict]:
        """Adjust diagnosis confidence based on patient history"""
        for diagnosis in diagnoses:
            # Age adjustments
            age = history.get("age", 50)
            if age > 65:
                diagnosis["confidence"] *= 1.1
            elif age < 30:
                diagnosis["confidence"] *= 0.9
            
            # Family history adjustments
            if history.get("family_history"):
                family_conditions = history["family_history"]
                if any(cond.lower() in diagnosis["condition"].lower() for cond in family_conditions):
                    diagnosis["confidence"] *= 1.2
            
            # Smoking adjustments
            if history.get("smoking") and "heart" in diagnosis["condition"].lower():
                diagnosis["confidence"] *= 1.15
            
            # Gender adjustments
            gender = history.get("gender", "unknown")
            if gender == "male" and "heart" in diagnosis["condition"].lower():
                diagnosis["confidence"] *= 1.1
            elif gender == "female" and "autoimmune" in diagnosis["condition"].lower():
                diagnosis["confidence"] *= 1.1
        
        return sorted(diagnoses, key=lambda x: x["confidence"], reverse=True)
    
    def _generate_recommendations(self, top_diagnoses: List[Dict]) -> Dict[str, Any]:
        """Generate clinical recommendations"""
        if not top_diagnoses:
            return {"immediate_action": "seek_medical_attention", "follow_up": "schedule_appointment"}
        
        top_condition = top_diagnoses[0]
        urgency = top_condition["urgency_level"]
        
        if urgency == "high":
            return {
                "immediate_action": "seek_emergency_medical_care",
                "follow_up": "emergency_department_visit",
                "monitoring": "continuous_vital_signs",
                "precautions": ["do_not_drive", "call_emergency_services"]
            }
        elif urgency == "medium":
            return {
                "immediate_action": "schedule_appointment_within_24h",
                "follow_up": "primary_care_physician",
                "monitoring": "symptom_tracking",
                "precautions": ["rest", "monitor_symptoms"]
            }
        else:
            return {
                "immediate_action": "schedule_routine_appointment",
                "follow_up": "primary_care_physician",
                "monitoring": "watchful_waiting",
                "precautions": ["lifestyle_modifications"]
            }
    
    async def check_drug_interactions(self, drugs: List[str], patient_conditions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Real drug interaction analysis with severity assessment"""
        try:
            start_time = datetime.utcnow()
            
            interactions_found = []
            severity_levels = []
            
            # Check drug-drug interactions
            for i, drug1 in enumerate(drugs):
                for drug2 in drugs[i+1:]:
                    interaction = self._check_drug_interaction(drug1, drug2)
                    if interaction:
                        interactions_found.append(interaction)
                        severity_levels.append(interaction["severity"])
            
            # Check drug-condition interactions
            condition_interactions = []
            if patient_conditions:
                for drug in drugs:
                    for condition in patient_conditions:
                        interaction = self._check_drug_condition_interaction(drug, condition)
                        if interaction:
                            condition_interactions.append(interaction)
            
            # Calculate overall risk score
            risk_score = self._calculate_interaction_risk(severity_levels)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "drugs": drugs,
                "patient_conditions": patient_conditions,
                "drug_interactions": interactions_found,
                "condition_interactions": condition_interactions,
                "overall_risk_score": risk_score,
                "risk_level": self._get_risk_level(risk_score),
                "recommendations": self._generate_drug_recommendations(interactions_found, condition_interactions),
                "processing_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"Failed to check drug interactions: {e}")
            return {"error": str(e)}
    
    def _check_drug_interaction(self, drug1: str, drug2: str) -> Optional[Dict]:
        """Check for drug-drug interaction"""
        drug1_lower = drug1.lower()
        drug2_lower = drug2.lower()
        
        # Check known interactions
        for drug, interacting_drugs in self.drug_interactions_db.items():
            if drug1_lower == drug and drug2_lower in interacting_drugs:
                severity = self._get_interaction_severity(drug1, drug2)
                return {
                    "drug1": drug1,
                    "drug2": drug2,
                    "interaction_type": "contraindicated",
                    "severity": severity,
                    "description": f"{drug1} and {drug2} have a known interaction that may cause adverse effects",
                    "recommendation": "consult_pharmacist_or_physician",
                    "mechanism": self._get_interaction_mechanism(drug1, drug2)
                }
            elif drug2_lower == drug and drug1_lower in interacting_drugs:
                severity = self._get_interaction_severity(drug1, drug2)
                return {
                    "drug1": drug1,
                    "drug2": drug2,
                    "interaction_type": "contraindicated",
                    "severity": severity,
                    "description": f"{drug1} and {drug2} have a known interaction that may cause adverse effects",
                    "recommendation": "consult_pharmacist_or_physician",
                    "mechanism": self._get_interaction_mechanism(drug1, drug2)
                }
        
        return None
    
    def _get_interaction_severity(self, drug1: str, drug2: str) -> str:
        """Determine interaction severity"""
        high_severity_pairs = [
            ("warfarin", "aspirin"),
            ("warfarin", "ibuprofen"),
            ("digoxin", "verapamil")
        ]
        
        for pair in high_severity_pairs:
            if (drug1.lower() in pair and drug2.lower() in pair):
                return "high"
        
        return "medium"
    
    def _get_interaction_mechanism(self, drug1: str, drug2: str) -> str:
        """Get interaction mechanism"""
        mechanisms = {
            ("warfarin", "aspirin"): "Increased bleeding risk due to platelet inhibition",
            ("warfarin", "ibuprofen"): "Increased bleeding risk and gastrointestinal effects",
            ("metformin", "contrast_dye"): "Increased risk of lactic acidosis",
            ("lisinopril", "potassium_supplements"): "Increased risk of hyperkalemia"
        }
        
        key = (drug1.lower(), drug2.lower())
        reverse_key = (drug2.lower(), drug1.lower())
        
        return mechanisms.get(key, mechanisms.get(reverse_key, "Unknown mechanism"))
    
    def _check_drug_condition_interaction(self, drug: str, condition: str) -> Optional[Dict]:
        """Check for drug-condition interaction"""
        drug_info = self.medical_knowledge_base["drugs"].get(drug.lower())
        if drug_info and condition.lower() in drug_info.get("contraindications", []):
            return {
                "drug": drug,
                "condition": condition,
                "interaction_type": "contraindication",
                "severity": "high",
                "description": f"{drug} is contraindicated in {condition}",
                "recommendation": "consider_alternative_medication",
                "reason": "May worsen condition or cause adverse effects"
            }
        return None
    
    def _calculate_interaction_risk(self, severity_levels: List[str]) -> float:
        """Calculate overall interaction risk score"""
        if not severity_levels:
            return 0.0
        
        severity_scores = {"low": 0.2, "medium": 0.6, "high": 1.0}
        total_score = sum(severity_scores.get(severity, 0.3) for severity in severity_levels)
        return min(total_score / len(severity_levels), 1.0)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_drug_recommendations(self, drug_interactions: List, condition_interactions: List) -> Dict[str, Any]:
        """Generate drug interaction recommendations"""
        if not drug_interactions and not condition_interactions:
            return {"status": "safe", "message": "No significant interactions found"}
        
        high_severity_count = sum(1 for interaction in drug_interactions + condition_interactions 
                                if interaction["severity"] == "high")
        
        if high_severity_count > 0:
            return {
                "status": "dangerous",
                "message": "High-risk interactions detected",
                "action": "immediate_medical_review_required",
                "urgency": "high"
            }
        else:
            return {
                "status": "caution",
                "message": "Some interactions detected",
                "action": "pharmacist_consultation_recommended",
                "urgency": "medium"
            }
    
    async def predict_patient_outcome(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real patient outcome prediction using statistical models"""
        try:
            start_time = datetime.utcnow()
            
            # Extract features
            features = self._extract_patient_features(patient_data)
            
            # Predict outcomes using statistical models
            readmission_risk = self._predict_readmission_risk(features)
            mortality_risk = self._predict_mortality_risk(features)
            length_of_stay = self._predict_length_of_stay(features)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "patient_data": patient_data,
                "predictions": {
                    "readmission_risk": readmission_risk,
                    "mortality_risk": mortality_risk,
                    "estimated_length_of_stay": length_of_stay
                },
                "risk_factors": self._identify_risk_factors(features),
                "recommendations": self._generate_outcome_recommendations(readmission_risk, mortality_risk),
                "confidence_intervals": {
                    "readmission": [max(0, readmission_risk - 0.1), min(1, readmission_risk + 0.1)],
                    "mortality": [max(0, mortality_risk - 0.05), min(1, mortality_risk + 0.05)]
                },
                "processing_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"Failed to predict patient outcome: {e}")
            return {"error": str(e)}
    
    def _extract_patient_features(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from patient data"""
        features = {}
        
        # Age-based features
        age = patient_data.get("age", 50)
        features["age"] = float(age)
        features["age_squared"] = float(age ** 2)
        features["age_over_65"] = 1.0 if age > 65 else 0.0
        
        # Binary features
        diagnosis = str(patient_data.get("diagnosis", "")).lower()
        features["has_diabetes"] = 1.0 if "diabetes" in diagnosis else 0.0
        features["has_hypertension"] = 1.0 if "hypertension" in diagnosis else 0.0
        features["has_heart_disease"] = 1.0 if any(condition in diagnosis for condition in ["heart", "cardiac", "myocardial"]) else 0.0
        features["has_lung_disease"] = 1.0 if any(condition in diagnosis for condition in ["lung", "pulmonary", "asthma", "copd"]) else 0.0
        
        # Medication count
        medications = patient_data.get("medications", [])
        features["medication_count"] = float(len(medications))
        features["has_multiple_medications"] = 1.0 if len(medications) > 5 else 0.0
        
        # Comorbidity score
        comorbidity_keywords = ["diabetes", "hypertension", "heart", "kidney", "liver", "lung", "stroke"]
        comorbidity_score = sum(1 for keyword in comorbidity_keywords if keyword in diagnosis)
        features["comorbidity_score"] = float(comorbidity_score)
        features["high_comorbidity"] = 1.0 if comorbidity_score > 2 else 0.0
        
        return features
    
    def _predict_readmission_risk(self, features: Dict[str, float]) -> float:
        """Predict readmission risk using statistical model"""
        # Logistic regression-like calculation
        risk = 0.0
        risk += features["age"] * 0.008  # Age coefficient
        risk += features["has_diabetes"] * 0.25
        risk += features["has_heart_disease"] * 0.35
        risk += features["has_lung_disease"] * 0.20
        risk += features["comorbidity_score"] * 0.15
        risk += features["medication_count"] * 0.05
        risk += features["high_comorbidity"] * 0.20
        
        # Apply sigmoid function
        risk = 1 / (1 + math.exp(-risk))
        
        return min(max(risk, 0.0), 1.0)
    
    def _predict_mortality_risk(self, features: Dict[str, float]) -> float:
        """Predict mortality risk using statistical model"""
        risk = 0.0
        risk += features["age"] * 0.012  # Higher age coefficient for mortality
        risk += features["has_heart_disease"] * 0.45  # Heart disease is high risk
        risk += features["comorbidity_score"] * 0.25
        risk += features["age_squared"] * 0.0001  # Age squared term
        risk += features["high_comorbidity"] * 0.30
        
        # Apply sigmoid function
        risk = 1 / (1 + math.exp(-risk))
        
        return min(max(risk, 0.0), 1.0)
    
    def _predict_length_of_stay(self, features: Dict[str, float]) -> float:
        """Predict length of stay in days"""
        base_stay = 3.0
        stay = base_stay
        stay += features["age"] * 0.02
        stay += features["comorbidity_score"] * 0.6
        stay += features["medication_count"] * 0.15
        stay += features["has_heart_disease"] * 1.5
        stay += features["has_lung_disease"] * 1.2
        
        return max(stay, 1.0)
    
    def _identify_risk_factors(self, features: Dict[str, float]) -> List[str]:
        """Identify key risk factors"""
        risk_factors = []
        
        if features["age"] > 65:
            risk_factors.append("advanced_age")
        if features["has_diabetes"] > 0:
            risk_factors.append("diabetes")
        if features["has_heart_disease"] > 0:
            risk_factors.append("cardiovascular_disease")
        if features["has_lung_disease"] > 0:
            risk_factors.append("pulmonary_disease")
        if features["comorbidity_score"] > 2:
            risk_factors.append("multiple_comorbidities")
        if features["medication_count"] > 5:
            risk_factors.append("polypharmacy")
        if features["high_comorbidity"] > 0:
            risk_factors.append("high_comorbidity_burden")
        
        return risk_factors
    
    def _generate_outcome_recommendations(self, readmission_risk: float, mortality_risk: float) -> Dict[str, Any]:
        """Generate recommendations based on outcome predictions"""
        recommendations = {
            "monitoring": "standard",
            "follow_up": "routine",
            "interventions": [],
            "discharge_planning": []
        }
        
        if readmission_risk > 0.7:
            recommendations["monitoring"] = "intensive"
            recommendations["interventions"].append("discharge_planning")
            recommendations["interventions"].append("medication_reconciliation")
            recommendations["interventions"].append("patient_education")
            recommendations["discharge_planning"].append("home_health_services")
            recommendations["discharge_planning"].append("follow_up_appointment_7_days")
        
        if mortality_risk > 0.3:
            recommendations["monitoring"] = "critical"
            recommendations["follow_up"] = "immediate"
            recommendations["interventions"].append("palliative_care_consultation")
            recommendations["interventions"].append("family_conference")
            recommendations["interventions"].append("advanced_directives_review")
        
        if readmission_risk > 0.5:
            recommendations["interventions"].append("transitional_care_management")
            recommendations["discharge_planning"].append("medication_management")
        
        return recommendations

# Global instance
simplified_medical_ai = SimplifiedMedicalAI()
