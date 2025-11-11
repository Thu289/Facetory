'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, SparklesIcon, EyeIcon, PhotoIcon } from '@heroicons/react/24/outline';
import { apiService, type StyleData } from '../services/api';
import RegionPreview from './RegionPreview';

interface StyleUploadProps {
  onUpload: (file: File, name?: string) => Promise<void>;
  isUploading?: boolean;
  onStyleCreated?: (style: StyleData) => void;  // Callback when style is created
}

export default function StyleUpload({ onUpload, isUploading = false, onStyleCreated }: StyleUploadProps) {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [styleFile, setStyleFile] = useState<File | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [previewFile, setPreviewFile] = useState<File | null>(null);
  const [useStyleImageForPreview, setUseStyleImageForPreview] = useState(true);
  const [name, setName] = useState('');
  const [createdStyle, setCreatedStyle] = useState<StyleData | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showMasks, setShowMasks] = useState(false);
  const [maskPreviews, setMaskPreviews] = useState<Record<string, string>>({});
  const [regionsDetected, setRegionsDetected] = useState<string[]>([]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setStyleFile(file);
    setUseStyleImageForPreview(true);
    setPreviewFile(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setUploadedImage(result);
      setPreviewImage(result);
    };
    reader.readAsDataURL(file);
  }, []);

  const onDropPreview = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setPreviewFile(file);
    setUseStyleImageForPreview(false);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
    },
    multiple: false,
  });

  const {
    getRootProps: getPreviewRootProps,
    getInputProps: getPreviewInputProps,
    isDragActive: isPreviewDragActive,
  } = useDropzone({
    onDrop: onDropPreview,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
    },
    multiple: false,
  });

  const handleSubmit = async () => {
    if (!styleFile) {
      alert('Please upload a style image first.');
      return;
    }

    try {
      const previewUploadFile = useStyleImageForPreview ? styleFile : previewFile ?? undefined;

      const style = await apiService.createStyle(styleFile, {
        name: name || undefined,
        previewFile: previewUploadFile,
      });
      setCreatedStyle(style);
      setShowPreview(true);
      setShowMasks(true);  // Auto-show masks after creation
      
      if (style.mask_previews) {
        setMaskPreviews(style.mask_previews);
      }
      if (style.regions_detected) {
        setRegionsDetected(style.regions_detected);
      }
      
      if (onStyleCreated) {
        onStyleCreated(style);
      }
      
      await onUpload(styleFile, name || undefined);
    } catch (error) {
      console.error('Upload error:', error);
      alert(`Failed to create style: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-6 md:grid-cols-2">
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isDragActive
              ? 'border-purple-500 bg-purple-50'
              : 'border-gray-300 hover:border-gray-400'
            }
          `}
        >
          <input {...getInputProps()} />
          <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          {isDragActive ? (
            <p className="text-purple-600">Drop the image here...</p>
          ) : (
            <div>
              <p className="text-gray-600 mb-2">
                Drag & drop the makeup reference (style) image
              </p>
              <p className="text-sm text-gray-500">
                Supports JPG, PNG up to 10MB
              </p>
            </div>
          )}
        </div>

        <div
          {...getPreviewRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isPreviewDragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
            }
          `}
        >
          <input {...getPreviewInputProps()} />
          <PhotoIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          {isPreviewDragActive ? (
            <p className="text-blue-600">Drop preview image...</p>
          ) : (
            <div>
              <p className="text-gray-600 mb-2">
                Optional: Drag & drop a model image for preview
              </p>
              <p className="text-sm text-gray-500">
                Leave empty to use the style image or the default preview
              </p>
            </div>
          )}
        </div>
      </div>

      {uploadedImage && (
        <div className="space-y-4">
          <div>
            <img
              src={uploadedImage}
              alt="Preview"
              className="max-w-full h-auto rounded-lg shadow-md"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Style Name (optional)
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Vintage Glam"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>

          <button
            onClick={handleSubmit}
            disabled={isUploading}
            className="w-full px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <SparklesIcon className="h-5 w-5" />
            {isUploading ? 'Creating Style...' : 'Create Style'}
          </button>
        </div>
      )}

      {/* Filter Preview Section */}
      <div className="flex items-center gap-3">
        <input
          id="use-style-preview"
          type="checkbox"
          className="h-4 w-4"
          checked={useStyleImageForPreview}
          onChange={(e) => {
            const checked = e.target.checked;
            setUseStyleImageForPreview(checked);
            if (checked && uploadedImage) {
              setPreviewImage(uploadedImage);
              setPreviewFile(null);
            }
          }}
          disabled={!styleFile}
        />
        <label htmlFor="use-style-preview" className="text-sm text-gray-700">
          Use style image as preview (if unchecked, default preview or uploaded preview image will be used)
        </label>
      </div>

      {previewImage && (
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-700">Preview Image</h3>
            <img
              src={previewImage}
              alt="Preview for filter"
              className="max-w-full h-auto rounded-lg shadow-md"
            />
          </div>
        </div>
      )}

      {createdStyle && createdStyle.metadata?.filter_preview && (
        <div className="mt-6 space-y-4 p-6 bg-purple-50 rounded-lg border border-purple-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center gap-2 text-purple-900">
              <EyeIcon className="h-5 w-5 text-purple-600" />
              Filter Preview
            </h3>
            <button
              onClick={() => setShowPreview(!showPreview)}
              className="px-3 py-1 text-sm bg-white rounded border border-purple-300 hover:bg-purple-100 text-purple-700"
            >
              {showPreview ? 'Hide' : 'Show'} Preview
            </button>
          </div>
          
          {showPreview && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {uploadedImage && (
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-700">Original Image</h4>
                    <div className="border-2 border-gray-300 rounded-lg overflow-hidden shadow">
                      <img
                        src={uploadedImage}
                        alt="Original"
                        className="w-full h-auto"
                        style={{ maxHeight: '400px', objectFit: 'contain' }}
                      />
                    </div>
                  </div>
                )}
                
                {createdStyle.metadata.filter_preview && (
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-700">With Filter Applied</h4>
                    <div className="border-2 border-purple-400 rounded-lg overflow-hidden shadow">
                      <img
                        src={createdStyle.metadata.filter_preview}
                        alt="Filter Preview"
                        className="w-full h-auto"
                        style={{ maxHeight: '400px', objectFit: 'contain' }}
                      />
                    </div>
                    <div className="text-sm text-gray-600">
                      <p><strong>Style:</strong> {createdStyle.name || createdStyle.style_id}</p>
                      <p><strong>Style ID:</strong> {createdStyle.style_id}</p>
                    </div>
                  </div>
                )}
              </div>


              <div className="pt-4 border-t border-purple-200">
                <p className="text-sm text-gray-600 mb-4">
                  ✅ Style created successfully! You can now use this filter in the camera or test it on other images.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {createdStyle && (
        <div className="space-y-4 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
          <RegionPreview style={createdStyle} />
        </div>
      )}

      {/* Segmentation Masks Section - Show after style creation (outside filter preview) */}
      {maskPreviews && Object.keys(maskPreviews).length > 0 && (
        <div className="mt-6 space-y-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-blue-900">
              Segmentation Masks (Sau khi xử lý)
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
              <p className="text-xs text-blue-600">
                💡 <strong>Color coding:</strong> Red = Lips, Blue = Eyes, Green = Eyebrows, Yellow = Nose, Gray = Skin/Face
              </p>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(maskPreviews).map(([regionName, maskDataUrl]) => (
                  <div key={regionName} className="space-y-2">
                    <h4 className="text-sm font-medium text-gray-700 capitalize">
                      {regionName} Mask
                    </h4>
                    <div className="border-2 border-blue-300 rounded-lg overflow-hidden shadow">
                      <img
                        src={maskDataUrl}
                        alt={`${regionName} mask`}
                        className="w-full h-auto"
                        style={{ maxHeight: '200px', objectFit: 'contain' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              
              {regionsDetected.length > 0 && (
                <div className="text-sm text-gray-600">
                  <p><strong>Regions detected:</strong> {regionsDetected.join(', ')}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

