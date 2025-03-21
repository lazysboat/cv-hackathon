# AI vs Real Image Detector

This tool uses a Hugging Face model to detect whether an image is AI-generated or real.

## Setup

### Using a Virtual Environment (Recommended)

#### Quick Setup with Convenience Scripts

**Windows Users:**
Simply double-click the `run_with_venv.bat` file to:
- Create a virtual environment (if needed)
- Activate the environment
- Be ready to run any of the detector commands

**macOS/Linux Users:**
1. Make the script executable (first time only):
   ```
   chmod +x run_with_venv.sh
   ```
2. Run the script:
   ```
   ./run_with_venv.sh
   ```

#### Manual Virtual Environment Setup

1. Make sure you have Python 3.7+ installed.

2. Run the setup script to create and initialize a virtual environment:
   ```bash
   python setup_venv.py
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     ai_detector_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source ai_detector_env/bin/activate
     ```

4. When you're done using the application, deactivate the virtual environment:
   ```
   deactivate
   ```

### Manual Installation (Without Virtual Environment)

If you prefer to install directly without a virtual environment:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. The first time you run the script, it will download the model from Hugging Face.

## Usage

### Graphical User Interface

For the most user-friendly experience, run the GUI application:

```bash
python gui_detector.py
```

This opens a window where you can:
- Select an image file using the "Select Image" button
- Analyze it with the "Analyze Image" button
- View the classification result and confidence score

### Single Image Classification (Command Line)

To classify a single image from the command line:

```bash
python ai_detector.py path/to/your/image.jpg
```

### Batch Processing (Command Line)

To process all images in a directory:

```bash
python batch_detector.py path/to/image/directory
```

You can also save the results to a CSV file:

```bash
python batch_detector.py path/to/image/directory --output results.csv
```

## Output

The tool will output:
- The classification result (Real or AI)
- The confidence score as a percentage

## Model Information

This tool uses the "Nahrawy/AIorNot" model from Hugging Face, which is designed to distinguish between AI-generated and real images.

## Requirements

- Python 3.7+
- All dependencies listed in requirements.txt