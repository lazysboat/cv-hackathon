import subprocess
import sys
import os
import platform

def create_venv():
    """Create and initialize a virtual environment for the AI detector project."""
    
    venv_name = "ai_detector_env"
    
    # Check if virtual environment directory already exists
    if os.path.exists(venv_name):
        print(f"Virtual environment '{venv_name}' already exists.")
        choice = input("Do you want to recreate it? (y/n): ").strip().lower()
        if choice != 'y':
            print("Using existing virtual environment.")
            return venv_name
        
        print(f"Removing existing '{venv_name}' directory...")
        if platform.system() == "Windows":
            subprocess.run(["rmdir", "/s", "/q", venv_name], shell=True)
        else:
            subprocess.run(["rm", "-rf", venv_name])
    
    # Create the virtual environment
    print(f"Creating virtual environment '{venv_name}'...")
    subprocess.run([sys.executable, "-m", "venv", venv_name])
    
    # Determine the pip path based on OS
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_name, "Scripts", "pip")
        python_path = os.path.join(venv_name, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_name, "bin", "pip")
        python_path = os.path.join(venv_name, "bin", "python")
    
    # Update pip
    print("Updating pip in the virtual environment...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"])
    
    # Install requirements
    print("Installing required packages from requirements.txt...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    
    print("\nVirtual environment setup complete!")
    print("\nTo activate the virtual environment:")
    
    if platform.system() == "Windows":
        print(f"    {venv_name}\\Scripts\\activate")
    else:
        print(f"    source {venv_name}/bin/activate")
    
    print("\nAfter activation, you can run:")
    print("    python ai_detector.py path/to/your/image.jpg")
    print("    python batch_detector.py path/to/directory")
    print("    python gui_detector.py")
    
    return venv_name

if __name__ == "__main__":
    create_venv() 