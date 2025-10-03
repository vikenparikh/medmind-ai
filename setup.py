#!/usr/bin/env python3
"""
NeuralVerse AI - Setup Script
The Ultimate AI Development Platform
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("="*80)
    print("🧠 NeuralVerse AI - Setup Script")
    print("   The Ultimate AI Development Platform")
    print("   Built by Viken Parikh")
    print("="*80)

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please install Python 3.7 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "logs",
        "data",
        "models",
        "uploads",
        "temp"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    print("\n🔧 Setting up environment configuration...")
    
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    env_content = """# NeuralVerse AI Configuration
APP_NAME=NeuralVerse AI
APP_VERSION=2.0.0
DEBUG=true
LOG_LEVEL=INFO

# OpenAI API Key (optional - some features require this)
OPENAI_API_KEY=

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000

# CORS Settings
CORS_ORIGINS=*
ALLOWED_HOSTS=*

# Database Configuration
DATABASE_URL=sqlite:///./neuralverse.db

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
        print("   💡 You can edit .env to customize settings")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def run_initial_tests():
    """Run initial tests to verify installation"""
    print("\n🧪 Running initial tests...")
    
    try:
        # Quick import test
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Test core imports
        import backend.app.main
        import backend.app.core.config
        import backend.app.services.ai_engine_mock
        
        print("✅ Core modules import successfully")
        
        # Run a quick test
        test_result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_neuralverse_ai.py::TestNeuralVerseAI::test_platform_root",
            "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=30)
        
        if test_result.returncode == 0:
            print("✅ Initial test passed")
        else:
            print("⚠️  Initial test failed, but setup can continue")
            print("   You can run tests later with: python test_runner.py")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Test execution failed: {e}")
        print("   You can run tests later with: python test_runner.py")
        return True

def show_completion_message():
    """Show setup completion message"""
    print("\n" + "="*80)
    print("🎉 NeuralVerse AI Setup Complete!")
    print("="*80)
    print()
    print("🚀 Quick Start:")
    print("   python start_neuralverse.py")
    print()
    print("🧪 Run Tests:")
    print("   python test_runner.py")
    print()
    print("📚 API Documentation:")
    print("   http://localhost:8000/docs (after starting server)")
    print()
    print("🌐 Frontend Interface:")
    print("   frontend/index.html (opens automatically)")
    print()
    print("🔧 Configuration:")
    print("   Edit .env file to customize settings")
    print()
    print("📖 Features Available:")
    print("   🤖 Multi-Agent AI Systems (CrewAI)")
    print("   🔍 Vector Search & Retrieval (LlamaIndex)")
    print("   🧠 Machine Learning Pipeline (PyTorch, Scikit-learn)")
    print("   📝 Natural Language Processing")
    print("   🎨 Computer Vision & Image Analysis")
    print("   🎵 Audio Processing & Speech Recognition")
    print("   🎯 Generative AI Content Creation")
    print("   📈 Real-time Analytics & Insights")
    print()
    print("💡 Built by Viken Parikh")
    print("   GitHub: https://github.com/vikenparikh")
    print("   LinkedIn: https://linkedin.com/in/vikenparikh")
    print("="*80)

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return 1
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed during directory creation")
        return 1
    
    # Create environment file
    if not create_env_file():
        print("❌ Setup failed during environment configuration")
        return 1
    
    # Run initial tests
    run_initial_tests()
    
    # Show completion message
    show_completion_message()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
