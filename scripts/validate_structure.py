#!/usr/bin/env python3
"""
Script to validate the project structure and imports
"""

import os
import sys
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_import(module_path, description):
    """Check if a module can be imported"""
    try:
        spec = importlib.util.find_spec(module_path)
        if spec is not None:
            print(f"✅ {description}: {module_path}")
            return True
        else:
            print(f"❌ {description}: {module_path} NOT IMPORTABLE")
            return False
    except Exception as e:
        print(f"❌ {description}: {module_path} ERROR - {e}")
        return False

def main():
    """Validate the project structure"""
    print("🔍 Validating Captcha Solve API Project Structure")
    print("=" * 50)
    
    # Check core files
    core_files = [
        ("app/__init__.py", "App package init"),
        ("app/main.py", "Main application"),
        ("app/api/__init__.py", "API package init"),
        ("app/api/endpoints.py", "API endpoints"),
        ("app/core/__init__.py", "Core package init"),
        ("app/core/config.py", "Configuration"),
        ("app/core/database.py", "Database manager"),
        ("app/models/__init__.py", "Models package init"),
        ("app/models/model.py", "PyTorch models"),
        ("app/models/predict.py", "Prediction logic"),
    ]
    
    print("\n📁 Core Application Files:")
    all_good = True
    for filepath, description in core_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check supporting files
    support_files = [
        ("scripts/start_server.py", "Server startup script"),
        ("examples/example_client.py", "Example client"),
        ("tests/test_api.py", "API tests"),
        ("tests/test_concurrency.py", "Concurrency tests"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation"),
        ("Dockerfile", "Docker config"),
        ("docker-compose.yml", "Docker Compose"),
    ]
    
    print("\n📋 Supporting Files:")
    for filepath, description in support_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check model file
    print("\n🤖 Model Files:")
    check_file_exists("model/model.pth", "Trained PyTorch model")
    
    # Check import structure (basic validation)
    print("\n🔧 Import Structure Validation:")
    imports_to_check = [
        ("app", "App package"),
        ("app.core", "Core package"),
        ("app.api", "API package"),
        ("app.models", "Models package"),
    ]
    
    for module_path, description in imports_to_check:
        check_import(module_path, description)
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Project structure validation completed successfully!")
        print("✅ All core files are in place")
        print("🚀 Ready to run: python scripts/start_server.py")
    else:
        print("⚠️  Some files are missing. Please check the structure.")
        
    print("\n📚 Project Structure:")
    print("├── app/              # Main application")
    print("├── scripts/          # Utility scripts")
    print("├── examples/         # Example code")
    print("├── tests/            # Test files")
    print("└── model/            # Trained models")

if __name__ == "__main__":
    main()
