import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload as UploadIcon, RefreshCw, ZoomIn, X } from 'lucide-react';
import ExtractionTable from './ExtractionTable';
import { TableData } from '../types';
import essenseLogo from '../assets/Essense labs logo.jpg';

const MAX_IMAGE_SIZE = 15 * 1024 * 1024; // 15 MB
const API_BASE_URL = 'http://127.0.0.1:5000';

// Security: Sanitize filename to prevent XSS
const sanitizeFilename = (filename: string): string => {
  // Remove any path traversal attempts and dangerous characters
  return filename
    .replace(/[^a-z0-9_\-\s.]/gi, '_')  // Replace special chars with underscore
    .replace(/\.\./g, '_')                // Remove path traversal
    .replace(/\s+/g, '_')                 // Replace spaces with underscore
    .replace(/^\.+|\.+$/g, '')            // Remove leading/trailing dots
    .substring(0, 255);                   // Limit length
};

// Security: Validate and sanitize image URL to prevent XSS
const sanitizeImageUrl = (url: string | null | undefined): string | null => {
  if (!url) return null;
  
  // Allow blob URLs (created by URL.createObjectURL)
  if (url.startsWith('blob:')) {
    return url;
  }
  
  // Allow data URLs for images only
  if (url.startsWith('data:image/')) {
    return url;
  }
  
  // Allow URLs from our API base URL
  if (url.startsWith(API_BASE_URL)) {
    try {
      const urlObj = new URL(url);
      // Only allow http/https protocols
      if (urlObj.protocol === 'http:' || urlObj.protocol === 'https:') {
        return url;
      }
    } catch {
      return null;
    }
  }
  
  // Reject any other URLs for security
  return null;
};

interface TyreData {
  image: File | null;
  dewarped: string | null;
  stitched: string | null;
  fallbackImage?: string | null;
  extractedData: any[];
  cropBoxes: CropBox[];
  isProcessing: boolean;
  isCompleted: boolean;
  perTextBoxes?: {
    label: string;
    ocr_text: string;
    confidence: number;
    bbox: number[];
    image_base64: string;
  }[];
  sessionId?: string;
}

interface CropBox {
  id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
}

interface DualTyreInterfaceProps {
  onCombinedResults?: (results: any) => void;
}

const DualTyreInterface: React.FC<DualTyreInterfaceProps> = ({ onCombinedResults }) => {
  const [topTyre, setTopTyre] = useState<TyreData>({
    image: null,
    dewarped: null,
    stitched: null,
    fallbackImage: null,
    extractedData: [],
    cropBoxes: [],
    isProcessing: false,
    isCompleted: false
  });

  const [bottomTyre, setBottomTyre] = useState<TyreData>({
    image: null,
    dewarped: null,
    stitched: null,
    fallbackImage: null,
    extractedData: [],
    cropBoxes: [],
    isProcessing: false,
    isCompleted: false
  });


  const [topImageType, setTopImageType] = useState<'thick' | 'light' | ''>('');
  const [bottomImageType, setBottomImageType] = useState<'thick' | 'light' | ''>('');
  const [zoomedImage, setZoomedImage] = useState<{ tyreType: 'top' | 'bottom', imageUrl: string } | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const topImageRef = useRef<HTMLImageElement>(null);
  const bottomImageRef = useRef<HTMLImageElement>(null);
  const topCanvasRef = useRef<HTMLCanvasElement>(null);
  const bottomCanvasRef = useRef<HTMLCanvasElement>(null);
  const zoomImageRef = useRef<HTMLImageElement>(null);
  const zoomCanvasRef = useRef<HTMLCanvasElement>(null);

  // Upload handling for individual tyres
  const handleImageLoad = (tyreType: 'top' | 'bottom', event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_IMAGE_SIZE) {
      alert('File size exceeds 15MB. Please select a smaller image.');
      event.target.value = '';
      return;
    }
    
    if (tyreType === 'top') {
      setTopTyre(prev => ({ ...prev, image: file }));
    } else {
      setBottomTyre(prev => ({ ...prev, image: file }));
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (tyreType: 'top' | 'bottom', e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > MAX_IMAGE_SIZE) {
        alert('File size exceeds 15MB. Please select a smaller image.');
        return;
      }
      
      if (tyreType === 'top') {
        setTopTyre(prev => ({ ...prev, image: file }));
      } else {
        setBottomTyre(prev => ({ ...prev, image: file }));
      }
    }
  };

  const buildTableFromDetections = (detections: Array<{ text?: string }>) => {
    const uniqueTexts = Array.from(
      new Set(
        detections
          .map(item => (item.text || '').trim())
          .filter(Boolean)
      )
    );

    const extractedData = uniqueTexts.map(text => ({
      text,
      parameter: text,
      confidence: 1
    }));

    const tableRows = uniqueTexts.map((text, index) => [String(index + 1), text]);

    return { extractedData, tableRows };
  };

  // Process individual tyre - Step 1: Upload and Dewarp
  const handleProcessTyre = async (tyreType: 'top' | 'bottom') => {
    const tyre = tyreType === 'top' ? topTyre : bottomTyre;
    if (!tyre.image) return;

    const imageType = tyreType === 'top' ? topImageType : bottomImageType;
    if (!imageType) {
      alert('Please select the image type (Thick or Light) before processing.');
      return;
    }

    const setTyre = tyreType === 'top' ? setTopTyre : setBottomTyre;
    setTyre(prev => ({ ...prev, isProcessing: true }));

    try {
      console.log('Starting upload for', tyreType, 'tyre');
      const formData = new FormData();
      formData.append('file', tyre.image);
      formData.append('image_type', imageType);
      formData.append('tyre_type', tyreType);
      if (activeSessionId) {
        formData.append('session_id', activeSessionId);
      }

      console.log('Sending request to backend...');
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      console.log('Response status:', response.status);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Upload failed:', errorText);
        throw new Error(`Upload failed: ${errorText}`);
      }

      const result = await response.json();

      const sessionId = result.session_id as string | undefined;
      if (sessionId) {
        setActiveSessionId(sessionId);
      }

      const cacheBuster = Date.now();
      const annotatedUrl = result.annotated_url ? `${API_BASE_URL}${result.annotated_url}?t=${cacheBuster}` : null;
      const fallbackDewarpUrl = result.dewarped_url ? `${API_BASE_URL}${result.dewarped_url}?t=${cacheBuster}` : null;
      const displayUrl = annotatedUrl || fallbackDewarpUrl;

      const detections = Array.isArray(result.detections) ? result.detections : [];
      const processed = buildTableFromDetections(detections);
      const extractedData = processed.extractedData;
      const tableRows = processed.tableRows;

      const detectionBoxes = detections
        .filter((item: any) => item && item.original_bbox)
        .map((item: any, index: number) => {
          const bbox = item.original_bbox || {};
          const leftRaw = bbox.left ?? 0;
          const topRaw = bbox.top ?? 0;
          const rightRaw = bbox.right ?? (leftRaw + (bbox.width ?? 0));
          const bottomRaw = bbox.bottom ?? (topRaw + (bbox.height ?? 0));
          const left = Math.round(leftRaw);
          const top = Math.round(topRaw);
          const right = Math.round(rightRaw);
          const bottom = Math.round(bottomRaw);
          return {
            label: item.chunk_name || `region_${index + 1}`,
            ocr_text: item.text || '',
            confidence: 1,
            bbox: [left, top, right, bottom],
            image_base64: ''
          };
        });
      
      setTyre(prev => ({
        ...prev,
        dewarped: displayUrl,
        stitched: annotatedUrl || fallbackDewarpUrl,
        fallbackImage: fallbackDewarpUrl,
        isProcessing: false,
        isCompleted: true,
        sessionId: sessionId,
        extractedData,
        cropBoxes: [],
        perTextBoxes: detectionBoxes
      }));

      if (displayUrl && zoomedImage && zoomedImage.tyreType === tyreType) {
        setZoomedImage({ tyreType, imageUrl: displayUrl });
      }

      if (tableRows.length > 0) {
        const tableData: TableData = {
          headers: tyreType === 'top'
            ? ['Serial Number', 'Top Tyre Data']
            : ['Serial Number', 'Bottom Tyre Data'],
          rows: tableRows
        };

        if (tyreType === 'top') {
          setTopTableData(tableData);
        } else {
          setBottomTableData(tableData);
        }
      }

      alert(`${tyreType === 'top' ? 'Top' : 'Bottom'} tyre uploaded and dewarped successfully!`);
    } catch (error) {
      console.error('Processing failed:', error);
      setTyre(prev => ({ ...prev, isProcessing: false }));
      alert('Processing failed. Please try again.');
    }
  };

  const handleImageError = (tyreType: 'top' | 'bottom') => {
    if (tyreType === 'top') {
      setTopTyre(prev => {
        if (prev.fallbackImage && prev.dewarped !== prev.fallbackImage) {
          if (zoomedImage && zoomedImage.tyreType === 'top') {
            setZoomedImage(current => current ? { ...current, imageUrl: prev.fallbackImage as string } : current);
          }
          return { ...prev, dewarped: prev.fallbackImage };
        }
        return prev;
      });
    } else {
      setBottomTyre(prev => {
        if (prev.fallbackImage && prev.dewarped !== prev.fallbackImage) {
          if (zoomedImage && zoomedImage.tyreType === 'bottom') {
            setZoomedImage(current => current ? { ...current, imageUrl: prev.fallbackImage as string } : current);
          }
          return { ...prev, dewarped: prev.fallbackImage };
        }
        return prev;
      });
    }
    setTimeout(() => redrawCanvas(tyreType), 150);
  };

  const handleCanvasMouseDown = (_e: React.MouseEvent<HTMLCanvasElement>, _tyreType: 'top' | 'bottom') => {
    // Cropping disabled in this workflow
  };

  const handleCanvasMouseMove = (_e: React.MouseEvent<HTMLCanvasElement>, _tyreType: 'top' | 'bottom') => {
    // Cropping disabled in this workflow
  };

  const handleCanvasMouseUp = (_e: React.MouseEvent<HTMLCanvasElement>, _tyreType: 'top' | 'bottom') => {
    // Cropping disabled in this workflow
  };

  // Redraw canvas
  const redrawCanvas = useCallback((tyreType: 'top' | 'bottom') => {
    const useZoom = zoomedImage && zoomedImage.tyreType === tyreType;
    const canvas = useZoom 
      ? zoomCanvasRef.current 
      : (tyreType === 'top' ? topCanvasRef.current : bottomCanvasRef.current);
    const image = useZoom 
      ? zoomImageRef.current 
      : (tyreType === 'top' ? topImageRef.current : bottomImageRef.current);

    if (!canvas || !image) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
  }, [zoomedImage]);

  // Initialize canvas when image loads
  const handleCanvasImageLoad = useCallback((tyreType: 'top' | 'bottom') => {
    const canvas = tyreType === 'top' ? topCanvasRef.current : bottomCanvasRef.current;
    const image = tyreType === 'top' ? topImageRef.current : bottomImageRef.current;

    if (!canvas || !image) return;

    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    const rect = image.getBoundingClientRect();
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    // Redraw canvas to show any existing crop boxes
    setTimeout(() => {
      redrawCanvas(tyreType);
    }, 100);
  }, [redrawCanvas]);

  // Redraw canvas when crop boxes change
  React.useEffect(() => {
    if (topTyre.dewarped) {
      setTimeout(() => {
        redrawCanvas('top');
      }, 100);
    }
  }, [topTyre.cropBoxes, topTyre.perTextBoxes, topTyre.dewarped, redrawCanvas]);

  React.useEffect(() => {
    if (bottomTyre.dewarped) {
      setTimeout(() => {
        redrawCanvas('bottom');
      }, 100);
    }
  }, [bottomTyre.cropBoxes, bottomTyre.perTextBoxes, bottomTyre.dewarped, redrawCanvas]);

  // Reset tyre data
  const resetTyre = (tyreType: 'top' | 'bottom') => {
    if (tyreType === 'top') {
      setTopTyre({
        image: null,
        dewarped: null,
        stitched: null,
        fallbackImage: null,
        extractedData: [],
        cropBoxes: [],
        isProcessing: false,
        isCompleted: false
      });
    } else {
      setBottomTyre({
        image: null,
        dewarped: null,
        stitched: null,
        fallbackImage: null,
        extractedData: [],
        cropBoxes: [],
        isProcessing: false,
        isCompleted: false
      });
    }
  };

  // Render individual tyre upload section (upload area only)
  const renderTyreUploadSection = (tyreType: 'top' | 'bottom', tyre: TyreData) => {
    const isTop = tyreType === 'top';

    return (
      <div className="bg-white rounded-lg overflow-hidden shadow-lg border border-gray-200">
        {/* Dark Header */}
        <div className="bg-gray-800 px-6 py-4">
          <h2 className="text-white font-bold text-lg text-center">
            {isTop ? 'Top Tyre Upload' : 'Bottom Tyre Upload'}
          </h2>
        </div>
        
        {/* Upload Area */}
        <div className="bg-white p-8">
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors relative ${
              'border-gray-300 hover:border-gray-400'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={(e) => handleDrop(tyreType, e)}
            onClick={() => !tyre.isProcessing && document.getElementById(`${tyreType}-file-input`)?.click()}
          >
            {tyre.isProcessing ? (
              <div className="space-y-4">
                <div className="flex justify-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                </div>
                <div>
                  <p className="text-lg font-medium text-gray-700">Processing...</p>
                  <p className="text-sm text-gray-500">Please wait while your image is being processed</p>
                </div>
              </div>
            ) : tyre.image ? (
              <div className="relative">
                <img 
                  src={URL.createObjectURL(tyre.image)} 
                  alt="Uploaded Preview" 
                  className="max-w-full max-h-64 mx-auto rounded-lg"
                />
                <div className="absolute inset-0 bg-black bg-opacity-50 rounded-lg flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                  <span className="text-white font-medium">Click to change image</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex justify-center">
                  <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                    <UploadIcon size={24} className="text-white" />
                  </div>
                </div>
                <div>
                  <p className="text-lg font-medium text-gray-700">Drop your image here or click to browse</p>
                  <p className="text-sm text-gray-500">Max 15MB • Supports: JPG, PNG, WEBP</p>
                </div>
              </div>
            )}
          </div>
          
          <div className="mt-6 flex flex-col items-center space-y-2">
            <label
              htmlFor={`${tyreType}-image-type`}
              className="text-sm font-semibold text-gray-700"
            >
              Select Image Type (required before processing)
            </label>
            <select
              id={`${tyreType}-image-type`}
              value={isTop ? topImageType : bottomImageType}
              onChange={(e) => {
                const newType = e.target.value as 'thick' | 'light' | '';
                if (isTop) {
                  setTopImageType(newType);
                } else {
                  setBottomImageType(newType);
                }
              }}
              className="w-full max-w-sm px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-gray-700"
            >
              <option value="">-- Select Image Type --</option>
              <option value="thick">Thick Image</option>
              <option value="light">Light Image</option>
            </select>
          </div>
          
          {/* Action Buttons */}
          <div className="flex justify-center mt-6 space-x-3">
            {tyre.image && !tyre.isProcessing && !tyre.isCompleted && (
              <button 
                className="flex items-center space-x-2 px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors shadow-md"
                onClick={() => handleProcessTyre(tyreType)}
              >
                <UploadIcon size={18} />
                <span>Process {isTop ? 'Top' : 'Bottom'} Tyre</span>
              </button>
            )}
            {!tyre.image && !tyre.isProcessing && (
              <label 
                htmlFor={`${tyreType}-file-input`} 
                className="flex items-center space-x-2 px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors cursor-pointer shadow-md"
              >
                <UploadIcon size={18} />
                <span>Load Image</span>
              </label>
            )}
            {tyre.image && (
              <button 
                className="flex items-center space-x-2 px-4 py-3 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
                onClick={() => resetTyre(tyreType)}
              >
                <RefreshCw size={18} />
                <span>Reset</span>
              </button>
            )}
          </div>
        </div>

        <input
          id={`${tyreType}-file-input`}
          type="file"
          accept="image/*"
          onChange={(e) => handleImageLoad(tyreType, e)}
          className="hidden"
        />
      </div>
    );
  };

  // Render dewarped image section separately (full width)
  const renderDewarpedSection = (tyreType: 'top' | 'bottom', tyre: TyreData) => {
    if (!tyre.dewarped) return null;

    const isTop = tyreType === 'top';
    const imageRef = isTop ? topImageRef : bottomImageRef;
    const canvasRef = isTop ? topCanvasRef : bottomCanvasRef;

    return (
      <div className="bg-white rounded-lg overflow-hidden shadow-lg border border-gray-200">
        {/* Dewarped Image Display */}
        {/* Stylish Black Header Bar */}
        <div className="bg-gray-900 px-6 py-4 border-b-4 border-white">
          <div className="flex justify-between items-center">
            <h3 className="text-xl font-bold text-white tracking-wide">
              {isTop ? 'Top Tyre - Annotated OCR Image' : 'Bottom Tyre - Annotated OCR Image'}
            </h3>
            {/* Image Type Display and Zoom Button */}
            <div className="flex items-center gap-3">
              <span className="text-sm font-semibold text-white">
                Image Type: { (isTop ? topImageType : bottomImageType) ? (isTop ? topImageType : bottomImageType)?.toUpperCase() : 'Not Selected' }
              </span>
              <button
                onClick={() => setZoomedImage({ tyreType, imageUrl: tyre.dewarped! })}
                className="flex items-center gap-2 px-4 py-2 bg-white text-gray-900 rounded-md text-sm font-bold transition-all hover:bg-gray-100 shadow-md"
              >
                <ZoomIn size={18} />
                <span>Zoom</span>
              </button>
            </div>
          </div>
        </div>
        
        {/* Image Content Area */}
        <div className="bg-gray-50 p-6">
          <div className="relative border border-gray-300 rounded-lg bg-white overflow-x-auto">
            <div className="relative inline-block">
              <img
                ref={imageRef}
                src={sanitizeImageUrl(tyre.dewarped) || ''}
                alt="Annotated OCR result"
                className="block"
                style={{ height: '60vh', width: 'auto' }}
                onLoad={() => handleCanvasImageLoad(tyreType)}
                onError={() => handleImageError(tyreType)}
              />
              <canvas
                ref={canvasRef}
                onMouseDown={(e) => handleCanvasMouseDown(e, tyreType)}
                onMouseMove={(e) => handleCanvasMouseMove(e, tyreType)}
                onMouseUp={(e) => handleCanvasMouseUp(e, tyreType)}
                className="absolute top-0 left-0"
                style={{
                  pointerEvents: 'auto'
                }}
              />
            </div>
          </div>

          <div className="mt-4 text-center text-sm text-gray-600">
            Automatic OCR is applied during upload. Manual cropping is disabled in this workflow.
          </div>
        </div>
      </div>
    );
  };

  // Separate state for top and bottom tables
  const [topTableData, setTopTableData] = useState<TableData>({
    headers: ['Serial Number', 'Top Tyre Data'],
    rows: []
  });

  const [bottomTableData, setBottomTableData] = useState<TableData>({
    headers: ['Serial Number', 'Bottom Tyre Data'],
    rows: []
  });

  // Track the last row count from each tyre to detect new OCR data
  useEffect(() => {
    if (topTyre.extractedData.length === 0) {
      setTopTableData(prev => ({ ...prev, rows: [] }));
      return;
    }

    const rows = topTyre.extractedData.map((item, index) => [
      String(index + 1),
                item.text || 'No text detected'
    ]);
    setTopTableData(prev => ({ ...prev, rows }));
  }, [topTyre.extractedData]);

  useEffect(() => {
    if (bottomTyre.extractedData.length === 0) {
      setBottomTableData(prev => ({ ...prev, rows: [] }));
      return;
    }

    const rows = bottomTyre.extractedData.map((item, index) => [
      String(index + 1),
      item.text || 'No text detected'
    ]);
    setBottomTableData(prev => ({ ...prev, rows }));
  }, [bottomTyre.extractedData]);



  // Handle save JSON - compare both tables by value
  const handleSaveJson = async () => {
    try {
      // Prompt user for JSON template name
      const templateName = prompt('Enter JSON Template Name:', 'Combined Tyre Data');
      
      // If user cancels or enters empty name, use default
      if (templateName === null) {
        return; // User cancelled
      }
      
      const finalTemplateName = templateName.trim() || 'Combined Tyre Data';
      
      // Gather unique values from both tables
      const topSet = new Set(
        topTableData.rows
          .map(row => row[1] ? row[1].trim() : '')
          .filter(Boolean)
      );
      
      const bottomSet = new Set(
        bottomTableData.rows
          .map(row => row[1] ? row[1].trim() : '')
          .filter(Boolean)
      );
      
      // Get all unique values from both tables
      const allValues = Array.from(new Set([...topSet, ...bottomSet]));
      
      if (allValues.length === 0) {
        alert('Please enter some data in the tables before saving!');
        return;
      }

      // Create comparison output
      const comparisonOutput = allValues.map((value, index) => ({
        serial: index + 1,
        value: value,
        sideA: topSet.has(value),   // true if in top table
        sideB: bottomSet.has(value)  // true if in bottom table
      }));

      console.log('Comparison output:', comparisonOutput);

      // Build the combined table data structure for backend
      const entries = comparisonOutput.map((item, index) => {
        const baseProps = [
          { Name: 'camera', Value: 'SideWall' },
          { Name: 'control_id', Value: '0' },
          { Name: 'engraving', Value: 'false' },
          { Name: 'expected_text', Value: item.value },
          { Name: 'instances', Value: '1' },
          { Name: 'mobile_stamp', Value: 'false' },
          { Name: 'name', Value: item.value },
          { Name: 'side_a', Value: item.sideA ? 'true' : 'false' },
          { Name: 'side_b', Value: item.sideB ? 'true' : 'false' },
          { Name: 'upside_down', Value: 'false' },
          { Name: 'user_id', Value: String(index + 1) }
        ];

        return {
          Properties: baseProps
        };
      });

      console.log('Sending entries to backend:', entries);

      const response = await fetch(`${API_BASE_URL}/api/combine-tire-json`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          Entries: entries,
          customName: finalTemplateName
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Backend processing failed');
      }

      const jsonResult = await response.json();
      
      // Create filename from template name (sanitize for filesystem and XSS prevention)
      const sanitizedName = sanitizeFilename(finalTemplateName).toLowerCase();
      const filename = `${sanitizedName}.json`;
      
      // Download the JSON file with custom name
      const blob = new Blob([JSON.stringify(jsonResult, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      // Security: Use sanitized filename to prevent XSS
      a.download = filename;
      // Security: Temporarily append to body, click, then immediately remove
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      alert(`JSON file "${filename}" generated and downloaded successfully!`);
      
      // Call the callback if provided
      if (onCombinedResults) {
        onCombinedResults(jsonResult);
      }
    } catch (error) {
      console.error('JSON generation failed:', error);
      alert('Failed to generate JSON: ' + (error instanceof Error ? error.message : String(error)));
    }
  };

  // Refresh website to start new tyre session (keeps backend sessions for debugging)
  const handleStartNewPair = () => {
    window.location.reload();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Clean Header with Logo Banner - Centered */}
      <header className="bg-black border-b-2 border-gray-800 shadow-lg relative">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-center relative">
            {/* Centered Logo */}
            <div className="flex items-center justify-center">
              <img 
                src={essenseLogo} 
                alt="ESSENSE LABS" 
                className="h-14 w-auto object-contain max-w-md"
              />
            </div>
            
            {/* Right side - Navigation buttons (absolute positioned) */}
            <div className="absolute right-4 flex items-center gap-4">
              <button
                onClick={handleStartNewPair}
                className="px-4 py-2 bg-white hover:bg-gray-100 text-black rounded-lg font-medium transition-colors shadow-md"
              >
                Start New Tyre Session
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">

        {/* Top and Bottom Tyre Upload Sections - Side by Side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {renderTyreUploadSection('top', topTyre)}
          {renderTyreUploadSection('bottom', bottomTyre)}
        </div>

        {/* Top Tyre Dewarped Image - Full Width */}
        {renderDewarpedSection('top', topTyre)}

        {/* Bottom Tyre Dewarped Image - Full Width */}
        {renderDewarpedSection('bottom', bottomTyre)}

         {/* Two Separate Tables: Top and Bottom */}
         <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
           <h2 className="text-xl font-bold text-gray-900 mb-4">Extracted Data</h2>
           
           {/* Two tables side by side */}
           <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
             {/* Top Tyre Data Table */}
             <div className="border border-gray-300 rounded-lg overflow-hidden shadow-sm">
               <div className="bg-gray-800 px-4 py-3 border-b border-gray-700">
                 <h3 className="text-white font-semibold text-sm uppercase tracking-wider">Top Tyre Data</h3>
               </div>
               <div className="max-h-96 overflow-y-auto bg-white">
                <ExtractionTable
                  data={topTableData}
                  onDataChange={setTopTableData}
                />
               </div>
             </div>

             {/* Bottom Tyre Data Table */}
             <div className="border border-gray-300 rounded-lg overflow-hidden shadow-sm">
               <div className="bg-gray-800 px-4 py-3 border-b border-gray-700">
                 <h3 className="text-white font-semibold text-sm uppercase tracking-wider">Bottom Tyre Data</h3>
               </div>
               <div className="max-h-96 overflow-y-auto bg-white">
                <ExtractionTable
                  data={bottomTableData}
                  onDataChange={setBottomTableData}
                />
               </div>
             </div>
           </div>

           {/* Single Save as JSON button for both tables */}
           <div className="flex justify-center">
             <button
               onClick={handleSaveJson}
               className="flex items-center gap-2 px-8 py-4 bg-gray-800 text-white rounded-lg font-medium hover:bg-gray-900 transition-all shadow-lg"
             >
               <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
               </svg>
               <span>Save as JSON</span>
             </button>
           </div>
         </div>

      </div>

      {/* Zoom Modal */}
      {zoomedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 p-4"
          onClick={() => setZoomedImage(null)}
        >
          <div 
            className="relative bg-white rounded-lg shadow-2xl max-w-[95vw] max-h-[95vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="sticky top-0 z-10 bg-gray-800 px-6 py-4 flex justify-between items-center rounded-t-lg">
              <h2 className="text-white font-bold text-xl">
                {zoomedImage.tyreType === 'top' ? 'Top' : 'Bottom'} Tyre - Zoomed View
              </h2>
              <button
                onClick={() => setZoomedImage(null)}
                className="text-white hover:text-gray-300 transition-colors"
              >
                <X size={28} />
              </button>
            </div>
            
            {/* Modal Body */}
            <div className="p-6">
              <div className="relative border border-gray-300 rounded-lg overflow-hidden bg-white">
                <img
                  ref={zoomImageRef}
                  src={sanitizeImageUrl(zoomedImage.imageUrl) || ''}
                  alt="Zoomed Annotated OCR"
                  className="w-full h-auto"
                  onLoad={() => {
                    const canvas = zoomCanvasRef.current;
                    const image = zoomImageRef.current;
                    if (!canvas || !image) return;
                    
                    canvas.width = image.naturalWidth;
                    canvas.height = image.naturalHeight;
                    canvas.style.width = `${image.width}px`;
                    canvas.style.height = `${image.height}px`;
                    
                    // Redraw canvas for the zoomed view
                    setTimeout(() => {
                      redrawCanvas(zoomedImage.tyreType);
                    }, 100);
                  }}
                  onError={() => handleImageError(zoomedImage.tyreType)}
                />
                <canvas
                  ref={zoomCanvasRef}
                  onMouseDown={(e) => handleCanvasMouseDown(e, zoomedImage.tyreType)}
                  onMouseMove={(e) => handleCanvasMouseMove(e, zoomedImage.tyreType)}
                  onMouseUp={(e) => handleCanvasMouseUp(e, zoomedImage.tyreType)}
                  className="absolute top-0 left-0"
                  style={{
                    pointerEvents: 'auto'
                  }}
                />
              </div>

              <div className="mt-6 text-center text-sm text-gray-600 space-y-4">
                <p>OCR data is captured automatically during upload. Manual crop-based OCR controls are disabled.</p>
                <button
                  onClick={() => setZoomedImage(null)}
                  className="px-4 py-2 rounded-lg font-medium transition-all shadow-sm bg-gray-700 text-white hover:bg-gray-800"
                >
                  Close Zoom View
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default DualTyreInterface;
