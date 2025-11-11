'use client';

import React, { useState } from 'react';
import { PhotoIcon, SparklesIcon } from '@heroicons/react/24/outline';

interface SegmentationPreviewProps {
  onImageUpload?: (image: File) => void;
}

export default function SegmentationPreview({ onImageUpload }: SegmentationPreviewProps) {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [segmentationResult, setSegmentationResult] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    console.log('📁 File selected:', file.name, file.type, file.size);
    
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      console.log('✅ Image loaded:', result.substring(0, 50) + '...');
      setUploadedImage(result);
      setSegmentationResult(null);
      setError(null);
    };
    reader.onerror = (e) => {
      console.error('❌ File read error:', e);
      setError('Failed to read image file');
    };
    reader.readAsDataURL(file);
    
    if (onImageUpload) {
      onImageUpload(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.target.files?.[0];
    console.log('📤 File input changed:', file);
    if (file) {
      handleFileSelect(file);
    } else {
      setError('No file selected');
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    console.log('📥 File dropped:', file);
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file);
    } else {
      setError('Please drop an image file');
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const runSegmentation = async () => {
    if (!uploadedImage) return;

    try {
      setIsProcessing(true);
      setError(null);
      setSegmentationResult(null);

      // Convert base64 to File
      const response = await fetch(uploadedImage);
      const blob = await response.blob();
      const file = new File([blob], 'test.jpg', { type: blob.type });

      // Call API
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const formData = new FormData();
      formData.append('file', file);

      console.log('🔄 Running BiSeNet segmentation...');

      const apiResponse = await fetch(`${API_BASE_URL}/api/face/makeup/style_extract`, {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json().catch(() => ({ detail: apiResponse.statusText }));
        throw new Error(errorData.detail || `API Error: ${apiResponse.status}`);
      }

      const result = await apiResponse.json();
      
      if (result.success && result.segmentation) {
        console.log('✅ Segmentation completed!', {
          model: result.processing_info?.model,
          device: result.processing_info?.device,
          classes: result.processing_info?.segmentation_classes
        });
        setSegmentationResult(result);
      } else {
        throw new Error('API returned invalid response');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to run segmentation';
      setError(errorMessage);
      console.error('❌ Segmentation error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <SparklesIcon className="h-6 w-6 text-purple-600" />
          Face Segmentation Preview (BiSeNet)
        </h2>
        
        <div className="space-y-4">
          {/* Upload Area */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors border-gray-300 hover:border-purple-400 hover:bg-purple-50"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              onClick={(e) => e.stopPropagation()}
            />
            <PhotoIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <div>
              <p className="text-gray-600 mb-2">
                Drag & drop an image here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Upload an image to see BiSeNet face segmentation
              </p>
              {uploadedImage && (
                <p className="text-sm text-green-600 mt-2">
                  ✓ Image loaded successfully
                </p>
              )}
            </div>
          </div>

          {/* Run Segmentation Button */}
          {uploadedImage && (
            <button
              onClick={runSegmentation}
              disabled={isProcessing}
              className="w-full px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <SparklesIcon className="h-5 w-5" />
              {isProcessing ? 'Running Segmentation...' : 'Run BiSeNet Segmentation'}
            </button>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800 font-medium">Error</p>
              <p className="text-red-600 text-sm mt-1">{error}</p>
            </div>
          )}

          {/* Processing Info */}
          {segmentationResult?.processing_info && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-blue-800 font-medium">Processing Info</p>
              <div className="text-blue-600 text-sm mt-1 space-y-1">
                <p>Model: {segmentationResult.processing_info.model}</p>
                <p>Device: {segmentationResult.processing_info.device}</p>
                <p>Segmentation Classes: {segmentationResult.processing_info.segmentation_classes}</p>
              </div>
            </div>
          )}

          {/* Results */}
          {segmentationResult && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Segmentation Results</h3>
              
              {/* Original Cropped Face */}
              {segmentationResult.segmentation?.original_cropped && (
                <div className="space-y-2">
                  <h4 className="font-medium text-gray-700">1. Original Cropped Face</h4>
                  <div className="border rounded-lg overflow-hidden shadow">
                    <img
                      src={segmentationResult.segmentation.original_cropped}
                      alt="Original Cropped Face"
                      className="w-full h-auto"
                      style={{ maxHeight: '500px', objectFit: 'contain' }}
                    />
                  </div>
                </div>
              )}

              {/* Colorized Mask */}
              {segmentationResult.segmentation?.colorized_mask && (
                <div className="space-y-2">
                  <h4 className="font-medium text-gray-700">2. Colorized Segmentation Mask</h4>
                  <p className="text-sm text-gray-500">
                    Each color represents a different facial region (19 attributes)
                  </p>
                  <div className="border rounded-lg overflow-hidden shadow">
                    <img
                      src={segmentationResult.segmentation.colorized_mask}
                      alt="Colorized Mask"
                      className="w-full h-auto"
                      style={{ maxHeight: '500px', objectFit: 'contain' }}
                    />
                  </div>
                </div>
              )}

              {/* Annotated Image */}
              {segmentationResult.segmentation?.annotated_image && (
                <div className="space-y-2">
                  <h4 className="font-medium text-gray-700">3. Annotated Image (Overlay)</h4>
                  <p className="text-sm text-gray-500">
                    Segmentation mask overlaid on original image with transparency
                  </p>
                  <div className="border rounded-lg overflow-hidden shadow">
                    <img
                      src={segmentationResult.segmentation.annotated_image}
                      alt="Annotated Image"
                      className="w-full h-auto"
                      style={{ maxHeight: '500px', objectFit: 'contain' }}
                    />
                  </div>
                </div>
              )}

              {/* Region Colors */}
              {segmentationResult.segmentation?.region_colors && (
                <div className="space-y-4">
                  <h4 className="font-medium text-gray-700">Region Colors (Average Colors)</h4>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                      {Object.entries(segmentationResult.segmentation.region_colors).map(([region_name, color_info]: [string, any]) => (
                        <div key={region_name} className="bg-white rounded-lg p-3 shadow-sm border">
                          <div className="flex items-center gap-3 mb-2">
                            {segmentationResult.segmentation?.region_color_previews?.[region_name] && (
                              <img 
                                src={segmentationResult.segmentation.region_color_previews[region_name]}
                                alt={region_name}
                                className="w-12 h-12 rounded border-2 border-gray-300"
                              />
                            )}
                            <div className="flex-1">
                              <p className="font-medium text-gray-800 capitalize">
                                {region_name.replace('_', ' ')}
                              </p>
                              <p className="text-xs text-gray-500">
                                {color_info.pixel_count?.toLocaleString() || 0} pixels
                              </p>
                            </div>
                          </div>
                          <div className="space-y-1 text-xs">
                            <div className="flex items-center gap-2">
                              <div 
                                className="w-6 h-6 rounded border-2 border-gray-400"
                                style={{ backgroundColor: color_info.hex || `rgb(${color_info.rgb?.join(',') || '200,200,200'})` }}
                              />
                              <div>
                                <p className="text-gray-600">
                                  RGB: ({color_info.rgb?.join(', ') || 'N/A'})
                                </p>
                                <p className="text-gray-500 font-mono">
                                  {color_info.hex || 'N/A'}
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Attributes List (Fallback if region_colors not available) */}
              {!segmentationResult.segmentation?.region_colors && segmentationResult.segmentation?.attributes && (
                <div className="space-y-2">
                  <h4 className="font-medium text-gray-700">Detected Attributes</h4>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                      {Object.entries(segmentationResult.segmentation.attributes).map(([key, value]: [string, any]) => (
                        <div key={key} className="flex items-center gap-2">
                          <div 
                            className="w-4 h-4 rounded border"
                            style={{ backgroundColor: `rgb(${value.color?.join(',') || '200,200,200'})` }}
                          />
                          <span className="text-gray-700">{key.replace('_', ' ')}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

