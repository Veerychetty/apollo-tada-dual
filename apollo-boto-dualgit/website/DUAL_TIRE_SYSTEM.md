# Dual Tire Processing System

This system implements a comprehensive dual tire analysis workflow that processes both top and bottom tire views separately, then combines the results for comprehensive analysis.

## System Architecture

### Backend Endpoints

1. **`/upload-top`** - Upload and process top tire image
   - Method: POST
   - Input: FormData with file
   - Output: Dewarped image, session ID, circle detection data

2. **`/upload-bottom`** - Upload and process bottom tire image
   - Method: POST
   - Input: FormData with file
   - Output: Dewarped image, session ID, circle detection data

3. **`/process-top-ocr`** - Run OCR analysis on top tire
   - Method: POST
   - Input: JSON with session_id and image_type
   - Output: OCR results, template, annotated image

4. **`/process-bottom-ocr`** - Run OCR analysis on bottom tire
   - Method: POST
   - Input: JSON with session_id and image_type
   - Output: OCR results, template, annotated image

5. **`/combine-results`** - Combine results from both tires
   - Method: POST
   - Input: JSON with top_results and bottom_results
   - Output: Combined analysis with merged templates

### Frontend Components

1. **DualTireProcessor** - Main component for dual tire upload and processing
   - Features:
     - Separate upload areas for top and bottom tires
     - Image type selection (light/thick) for each tire
     - Real-time processing status
     - Visual feedback during processing

2. **Processing Workflow**:
   - Upload both tire images
   - Select image types (light/thick) for each tire
   - Process both tires through backend endpoints
   - Combine results for comprehensive analysis

## Key Features

### Dual Tire Processing
- **Separate Processing**: Each tire is processed independently
- **Image Type Selection**: Choose between 'light' and 'thick' image types for each tire
- **Session Management**: Each tire gets its own session for tracking
- **Combined Results**: Results are merged for comprehensive analysis

### Image Processing Pipeline
1. **Upload**: Images are uploaded to separate endpoints
2. **Dewarping**: Circle detection and dewarping for each tire
3. **OCR Processing**: Text extraction with configurable image types
4. **Result Combination**: Merging of results from both tires

### Data Structure
```json
{
  "top_tire": {
    "ocr_results": [...],
    "template": {...},
    "annotated_image": "base64...",
    "total_regions": 5,
    "session_id": "abc123_top"
  },
  "bottom_tire": {
    "ocr_results": [...],
    "template": {...},
    "annotated_image": "base64...",
    "total_regions": 3,
    "session_id": "def456_bottom"
  },
  "combined_analysis": {
    "total_regions": 8,
    "combined_ocr_results": [...],
    "analysis_summary": {
      "top_tire_regions": 5,
      "bottom_tire_regions": 3,
      "total_text_extracted": 8
    }
  },
  "combined_template": {
    "Name": "Combined Tire Analysis - 2024-01-15 10:30:00",
    "Top_Tire": {...},
    "Bottom_Tire": {...},
    "Combined_Entries": [...]
  }
}
```

## Usage

### Starting the System

1. **Backend**:
   ```bash
   cd backend
   python paddlever2.py
   ```

2. **Frontend**:
   ```bash
   cd apollo-tadaa-main
   npm run dev
   ```

### Processing Workflow

1. **Upload Images**: Drag and drop or click to upload both top and bottom tire images
2. **Select Image Types**: Choose 'light' or 'thick' for each tire based on image characteristics
3. **Process**: Click "Process Both Tires" to start the analysis
4. **View Results**: Combined results are displayed with comprehensive analysis

### API Usage

#### Upload Top Tire
```javascript
const formData = new FormData();
formData.append('file', topTireFile);

const response = await fetch('http://127.0.0.1:5000/upload-top', {
  method: 'POST',
  body: formData
});
```

#### Process OCR for Top Tire
```javascript
const response = await fetch('http://127.0.0.1:5000/process-top-ocr', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: 'abc123_top',
    image_type: 'light'
  })
});
```

#### Combine Results
```javascript
const response = await fetch('http://127.0.0.1:5000/combine-results', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    top_results: topOcrData,
    bottom_results: bottomOcrData
  })
});
```

## Benefits

1. **Comprehensive Analysis**: Both tire views are analyzed for complete coverage
2. **Flexible Processing**: Different image types can be selected for each tire
3. **Modular Design**: Each tire is processed independently
4. **Combined Insights**: Results are merged for comprehensive analysis
5. **Session Tracking**: Each tire maintains its own processing session

## Error Handling

- **Upload Failures**: Clear error messages for failed uploads
- **Processing Errors**: Graceful handling of OCR failures
- **Network Issues**: Retry mechanisms for API calls
- **Validation**: Input validation for image types and session IDs

## Future Enhancements

1. **Batch Processing**: Process multiple tire sets simultaneously
2. **Advanced Analytics**: Machine learning-based tire analysis
3. **Export Options**: Multiple export formats for results
4. **Real-time Updates**: WebSocket-based real-time processing updates
5. **Cloud Integration**: Cloud storage for processed images and results
