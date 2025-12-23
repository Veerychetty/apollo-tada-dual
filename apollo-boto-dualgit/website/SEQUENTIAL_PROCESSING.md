# Sequential Tire Processing System

This system implements **sequential processing** of tires - processing one tire completely (upload, dewarp, crop, OCR) before moving to the next tire, then combining the results.

## Processing Workflow

### Step 1: Upload Images
- Upload both top and bottom tire images
- Select image type (light/thick) for each tire
- Visual confirmation of uploaded images

### Step 2: Process Top Tire
- **Upload**: Send top tire image to backend
- **Dewarp**: Detect circle and dewarp the image
- **Crop**: Extract relevant regions
- **OCR**: Run text extraction with selected image type
- **Complete**: Top tire processing is fully complete

### Step 3: Process Bottom Tire
- **Upload**: Send bottom tire image to backend
- **Dewarp**: Detect circle and dewarp the image
- **Crop**: Extract relevant regions
- **OCR**: Run text extraction with selected image type
- **Complete**: Bottom tire processing is fully complete

### Step 4: Combine Results
- Merge OCR results from both tires
- Create combined template
- Generate comprehensive analysis

## Backend API

### Single Tire Processing Endpoint
```
POST /process-single-tire
```

**Form Data:**
- `file`: Image file
- `tire_type`: 'top' or 'bottom'
- `image_type`: 'light' or 'thick'

**Response:**
```json
{
  "tire_type": "top",
  "image_type": "light",
  "session_id": "abc123_top",
  "ocr_results": [...],
  "template": {...},
  "annotated_image": "base64...",
  "dewarped_image": "base64...",
  "overlay_image": "base64...",
  "total_regions": 5,
  "circle_detected": true,
  "center": [cx, cy],
  "radius": r
}
```

### Combine Results Endpoint
```
POST /combine-results
```

**JSON Body:**
```json
{
  "top_results": { /* top tire data */ },
  "bottom_results": { /* bottom tire data */ }
}
```

## Frontend Components

### SequentialTireProcessor
- **Visual Step Indicator**: Shows current processing step
- **Dual Upload Areas**: Separate areas for top and bottom tires
- **Image Type Selection**: Choose light/thick for each tire
- **Real-time Status**: Live updates during processing
- **Progress Tracking**: Visual feedback for each step

### Processing Steps UI
1. **Upload Images** - Initial state
2. **Process Top Tire** - Active when processing top tire
3. **Process Bottom Tire** - Active when processing bottom tire
4. **Combine Results** - Active when merging results

## Key Benefits

### Sequential Processing
- **Complete Processing**: Each tire is fully processed before moving to the next
- **Error Isolation**: Issues with one tire don't affect the other
- **Resource Management**: Better memory and CPU usage
- **Clear Progress**: User sees exactly which step is running

### Individual Tire Analysis
- **Independent Processing**: Each tire gets its own session
- **Custom Configuration**: Different image types for each tire
- **Complete Workflow**: Upload → Dewarp → Crop → OCR for each tire
- **Separate Results**: Individual results before combination

### Combined Analysis
- **Merged Templates**: Combined data structure
- **Comprehensive View**: Both tires analyzed together
- **Source Tracking**: Know which tire each result came from
- **Total Statistics**: Combined metrics and counts

## Usage Example

### Frontend Processing Flow
```javascript
// Step 1: Process top tire
const topFormData = new FormData();
topFormData.append('file', topImage.file);
topFormData.append('tire_type', 'top');
topFormData.append('image_type', 'light');

const topResponse = await fetch('/process-single-tire', {
  method: 'POST',
  body: topFormData
});

// Step 2: Process bottom tire
const bottomFormData = new FormData();
bottomFormData.append('file', bottomImage.file);
bottomFormData.append('tire_type', 'bottom');
bottomFormData.append('image_type', 'thick');

const bottomResponse = await fetch('/process-single-tire', {
  method: 'POST',
  body: bottomFormData
});

// Step 3: Combine results
const combineResponse = await fetch('/combine-results', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    top_results: topData,
    bottom_results: bottomData
  })
});
```

## Data Flow

```
User Uploads Images
        ↓
Select Image Types
        ↓
Process Top Tire (Complete)
        ↓
Process Bottom Tire (Complete)
        ↓
Combine Results
        ↓
Display Combined Analysis
```

## Error Handling

- **Individual Tire Errors**: If one tire fails, the other can still be processed
- **Network Issues**: Retry mechanisms for API calls
- **Validation**: Input validation for image types and files
- **User Feedback**: Clear error messages and recovery options

## Performance Benefits

1. **Memory Efficiency**: Process one tire at a time
2. **CPU Optimization**: Sequential processing reduces peak load
3. **Error Recovery**: Can retry individual tires
4. **User Experience**: Clear progress indication
5. **Resource Management**: Better backend resource utilization

## Future Enhancements

1. **Parallel Processing Option**: Allow parallel processing for faster results
2. **Batch Processing**: Process multiple tire sets
3. **Advanced Analytics**: ML-based tire analysis
4. **Real-time Updates**: WebSocket-based progress updates
5. **Cloud Processing**: Offload processing to cloud services
