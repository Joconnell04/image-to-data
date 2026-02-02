# Image to Data

Extract structured text from screenshots using OpenAI vision models. Capture any area of your screen and get clean, formatted text output—perfect for tables, charts, diagrams, and more.

## Features

- **Interactive Screen Capture**: Select any area of your screen with the macOS screenshot tool
- **Smart Mode Detection**: Automatically detects if your image contains a table, chart, diagram, or general content
- **Structured Output**: Extracts data in formats optimized for each content type:
  - **Tables**: TSV format (paste directly into spreadsheets)
  - **Charts**: Axis labels, legend, and data points
  - **Diagrams**: Components, labels, and relationships
  - **General**: OCR-like text extraction with key details

## Requirements

- **OpenAI API Key**: You need an OpenAI API key with access to vision models (GPT-4 Vision or newer)

## Setup

1. Install the extension
2. Run "Extract Info" for the first time
3. Enter your OpenAI API key when prompted in preferences

## Usage

1. Open Raycast and run **Extract Info**
2. Your screen will dim—click and drag to select the area you want to extract
3. Wait for the AI to process the image
4. The extracted text is automatically copied to your clipboard

## Preferences

| Preference | Description |
|------------|-------------|
| **OpenAI API Key** | Your OpenAI API key for vision model access |
| **Router Model** | Model used to classify the image type (table/chart/diagram/general) |
| **Extractor Model** | Model used for the actual data extraction |
| **Max Output Length** | Maximum characters in the output (default: 12000) |
| **Include Confidence** | Add a confidence score line to the output |
| **Force Mode** | Override automatic detection with a specific mode |
| **Debug Logging** | Enable verbose console logging for troubleshooting |

## Tips

- For best results with tables, ensure the entire table is visible in your selection
- Charts work best when axis labels and legends are clearly visible
- Press **Escape** to cancel the screenshot selection