#!/usr/bin/env python3
"""
MedMind AI - Medical Knowledge Intelligence Platform
Startup script for the MedMind AI server
"""

import os
import sys
import subprocess
import webbrowser
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        logger.info("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def start_server():
    """Start the MedMind AI server"""
    logger.info("🏥 Starting MedMind AI - Medical Knowledge Intelligence Platform")
    
    # Check if we're in the right directory
    if not Path("backend/app/main.py").exists():
        logger.error("❌ Please run this script from the MedMind-AI directory")
        return False
    
    # Start the server
    try:
        logger.info("🌐 Starting FastAPI server on http://localhost:8000")
        logger.info("📚 API Documentation will be available at http://localhost:8000/docs")
        logger.info("🏥 MedMind AI Medical Intelligence Platform is starting...")
        
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend.app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8000")
            logger.info("🌐 Opened MedMind AI in your default browser")
        except:
            logger.info("🌐 Please open http://localhost:8000 in your browser")
        
        logger.info("🎯 MedMind AI is now running!")
        logger.info("📊 Features available:")
        logger.info("  🏥 Clinical Decision Support")
        logger.info("  🧬 Medical Knowledge Graph") 
        logger.info("  🔬 Research Synthesis")
        logger.info("  💊 Drug Interaction Analysis")
        logger.info("  🩺 Symptom Analysis")
        logger.info("  📋 Differential Diagnosis")
        logger.info("  📊 Patient Outcome Prediction")
        logger.info("  📚 Medical Literature Search")
        logger.info("  🎯 Treatment Optimization")
        logger.info("  ⚡ Real-time Clinical Support")
        
        logger.info("\n🛑 Press Ctrl+C to stop the server")
        
        # Wait for user to stop
        try:
            server_process.wait()
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping MedMind AI server...")
            server_process.terminate()
            server_process.wait()
            logger.info("✅ MedMind AI server stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        return False

def main():
    """Main function"""
    print("=" * 80)
    print("🏥 MedMind AI - Medical Knowledge Intelligence Platform")
    print("=" * 80)
    print("Revolutionizing healthcare through intelligent medical knowledge synthesis")
    print("and clinical decision support.")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start server
    if not start_server():
        sys.exit(1)
    
    print("\n🎉 Thank you for using MedMind AI!")
    print("🏥 Your medical knowledge is now intelligent and connected!")

if __name__ == "__main__":
    main()
