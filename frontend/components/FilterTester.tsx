'use client';

import React, { useState, useCallback, useRef } from 'react';
import { CloudArrowUpIcon, SparklesIcon } from '@heroicons/react/24/outline';
import { apiService, type StyleData } from '../services/api';

interface FilterTesterProps {
  style: StyleData;
}

export default function FilterTester({ style }: FilterTesterProps) {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [filteredImage, setFilteredImage] = useState<string | null>(null);
  const [bisenetPreviews, setBisenetPreviews] = useState<Record<string, string>>({});
  const [maskPreviews, setMaskPreviews] = useState<Record<string, string>>({});
  const [regionsDetected, setRegionsDetected] = useState<string[]>([]);
  const [showBisenet, setShowBisenet] = useState(false);
  const [showMasks, setShowMasks] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback(async (file: File) => {
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
      setFilteredImage(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const applyFilter = async () => {
    if (!uploadedImage) return;

    try {
      setIsProcessing(true);
      setError(null);
      setFilteredImage(null); // Clear previous result

      // Convert base64 to File
      const response = await fetch(uploadedImage);
      const blob = await response.blob();
      const file = new File([blob], 'test.jpg', { type: blob.type });

      // Call API
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const formData = new FormData();
      formData.append('file', file);
      formData.append('style_id', style.style_id);
      console.log('🔄 Applying filter...', {
        style_id: style.style_id,
        file_name: file.name,
        file_size: file.size
      });

      const apiResponse = await fetch(`${API_BASE_URL}/api/makeup/style/apply_filter`, {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json().catch(() => ({ detail: apiResponse.statusText }));
        throw new Error(errorData.detail || `API Error: ${apiResponse.status}`);
      }

      const result = await apiResponse.json();
      
      if (result.success && result.filtered_image) {
        console.log('✅ Filter applied successfully!', {
          style_id: result.style_id,
          has_image: !!result.filtered_image,
          bisenet_previews: result.bisenet_previews,
          mask_previews: result.mask_previews,
          regions_detected: result.regions_detected
        });
        setFilteredImage(result.filtered_image);
        setBisenetPreviews(result.bisenet_previews || {});
        setMaskPreviews(result.mask_previews || {});
        setRegionsDetected(result.regions_detected || []);
        setShowBisenet(true); // Auto-show BiSeNet results
        setShowMasks(true); // Auto-show masks when available
      } else {
        throw new Error('API returned invalid response');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to apply filter';
      setError(errorMessage);
      console.error('❌ Filter application error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Test Filter on Image</h2>
        
        <div className="space-y-4">
          {/* Upload Area */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors border-gray-300 hover:border-purple-400 hover:bg-purple-50"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
            <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <div>
              <p className="text-gray-600 mb-2">
                Drag & drop an image here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Upload an image to test the filter
              </p>
            </div>
          </div>

          {/* Apply Button */}
          {uploadedImage && (
            <button
              onClick={applyFilter}
              disabled={isProcessing}
              className="w-full px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <SparklesIcon className="h-5 w-5" />
              {isProcessing ? 'Applying Filter...' : 'Apply Filter'}
            </button>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">{error}</p>
            </div>
          )}

          {/* Results */}
          {(uploadedImage || filteredImage) && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Comparison</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {uploadedImage && (
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-700">Original Image</h4>
                    <div className="relative border rounded-lg overflow-hidden shadow">
                      <img
                        src={uploadedImage}
                        alt="Original"
                        className="w-full h-auto"
                        style={{ maxHeight: '500px', objectFit: 'contain' }}
                      />
                    </div>
                  </div>
                )}
                {filteredImage ? (
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-700">Filtered Image</h4>
                    <div className="relative border rounded-lg overflow-hidden shadow">
                      <img
                        src={filteredImage}
                        alt="Filtered"
                        className="w-full h-auto"
                        style={{ maxHeight: '500px', objectFit: 'contain' }}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-700">Filtered Image</h4>
                    <div className="border rounded-lg p-8 text-center text-gray-400 bg-gray-50" style={{ minHeight: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {isProcessing ? 'Processing...' : 'Click "Apply Filter" to see result'}
                    </div>
                  </div>
                )}
              </div>

              {/* BiSeNet Results Preview */}
              {Object.keys(bisenetPreviews).length > 0 && (
                <div className="mt-6 space-y-4 p-4 bg-green-50 rounded-lg border border-green-200">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-green-900">
                      BiSeNet Segmentation Results
                    </h3>
                    <button
                      onClick={() => setShowBisenet(!showBisenet)}
                      className="px-3 py-1 text-sm bg-white rounded border border-green-300 hover:bg-green-100 text-green-700"
                    >
                      {showBisenet ? 'Hide' : 'Show'} BiSeNet Results
                    </button>
                  </div>

                  {showBisenet && (
                    <div className="space-y-6">
                      {/* Raw BiSeNet Results (before processing) */}
                      {(bisenetPreviews.raw_segmentation_mask || bisenetPreviews.raw_colorized_mask) && (
                        <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <h4 className="font-semibold text-blue-900 mb-3">
                            📊 Raw BiSeNet Results (Trước khi xử lý)
                          </h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {bisenetPreviews.raw_segmentation_mask && (
                              <div className="space-y-2">
                                <h5 className="text-sm font-medium text-gray-700">
                                  Raw Segmentation Mask (Class IDs)
                                </h5>
                                <p className="text-xs text-gray-500">
                                  Mỗi pixel có giá trị class_id (0-19), hiển thị dạng grayscale
                                </p>
                                <div className="border rounded-lg overflow-hidden shadow">
                                  <img
                                    src={bisenetPreviews.raw_segmentation_mask}
                                    alt="Raw Segmentation Mask"
                                    className="w-full h-auto"
                                    style={{ maxHeight: '400px', objectFit: 'contain' }}
                                  />
                                </div>
                              </div>
                            )}
                            {bisenetPreviews.raw_colorized_mask && (
                              <div className="space-y-2">
                                <h5 className="text-sm font-medium text-gray-700">
                                  Raw Colorized Mask (Gốc từ BiSeNet)
                                </h5>
                                <p className="text-xs text-gray-500">
                                  Colorized mask gốc ở kích thước 512x512, chưa resize
                                </p>
                                <div className="border rounded-lg overflow-hidden shadow">
                                  <img
                                    src={bisenetPreviews.raw_colorized_mask}
                                    alt="Raw Colorized Mask"
                                    className="w-full h-auto"
                                    style={{ maxHeight: '400px', objectFit: 'contain' }}
                                  />
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Processed BiSeNet Results (after resizing/blending) */}
                      {(bisenetPreviews.colorized_mask || bisenetPreviews.annotated_image) && (
                        <div className="p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                          <h4 className="font-semibold text-yellow-900 mb-3">
                            🎨 Processed BiSeNet Results (Sau khi xử lý)
                          </h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {bisenetPreviews.colorized_mask && (
                              <div className="space-y-2">
                                <h5 className="text-sm font-medium text-gray-700">
                                  Colorized Mask (Resized)
                                </h5>
                                <p className="text-xs text-gray-500">
                                  Colorized mask đã resize về kích thước ảnh gốc để tạo annotated image
                                </p>
                                <div className="border rounded-lg overflow-hidden shadow">
                                  <img
                                    src={bisenetPreviews.colorized_mask}
                                    alt="Colorized Mask"
                                    className="w-full h-auto"
                                    style={{ maxHeight: '400px', objectFit: 'contain' }}
                                  />
                                </div>
                              </div>
                            )}
                            {bisenetPreviews.annotated_image && (
                              <div className="space-y-2">
                                <h5 className="text-sm font-medium text-gray-700">
                                  Annotated Image (Blended)
                                </h5>
                                <p className="text-xs text-gray-500">
                                  60% ảnh gốc + 40% colorized mask (resized)
                                </p>
                                <div className="border rounded-lg overflow-hidden shadow">
                                  <img
                                    src={bisenetPreviews.annotated_image}
                                    alt="Annotated Image"
                                    className="w-full h-auto"
                                    style={{ maxHeight: '400px', objectFit: 'contain' }}
                                  />
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Region Masks Section */}
              {Object.keys(maskPreviews).length > 0 && (
                <div className="mt-6 space-y-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-blue-900">
                      Segmentation Masks (Debug)
                    </h3>
                    <button
                      onClick={() => setShowMasks(!showMasks)}
                      className="px-3 py-1 text-sm bg-white rounded border border-blue-300 hover:bg-blue-100 text-blue-700"
                    >
                      {showMasks ? 'Hide' : 'Show'} Masks
                    </button>
                  </div>

                  {showMasks && (
                    <div className="space-y-4">
                      {regionsDetected.length > 0 && (
                        <p className="text-sm text-blue-800">
                          <strong>Regions detected:</strong> {regionsDetected.join(', ')}
                        </p>
                      )}

                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {Object.entries(maskPreviews).map(([regionName, maskImage]) => (
                          <div key={regionName} className="space-y-2">
                            <h4 className="font-medium text-gray-700 capitalize">
                              {regionName === 'face' ? 'Face Mask' : `${regionName} Mask`}
                            </h4>
                            <div className="border rounded-lg overflow-hidden shadow">
                              <img
                                src={maskImage}
                                alt={`${regionName} mask`}
                                className="w-full h-auto"
                                style={{ maxHeight: '250px', objectFit: 'contain' }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="pt-2 border-t border-blue-200">
                        <p className="text-xs text-blue-600">
                          💡 <strong>Color coding:</strong> Red = Lips, Blue = Eyes, Green = Eyebrows, Yellow = Nose, Gray = Skin/Face
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

