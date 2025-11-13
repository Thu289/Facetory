'use client';

import React, { useState } from 'react';
import dynamic from 'next/dynamic';

// Dynamic import with SSR disabled for browser-only components
const CameraFilter = dynamic(() => import('./CameraFilter'), {
  ssr: false,
});
import StyleSelector from './StyleSelector';
import { apiService, type StyleData } from '../services/api';
import StyleUpload from './StyleUpload';
import FilterTester from './FilterTester';

export default function FilterPageContent() {
  const [selectedStyle, setSelectedStyle] = useState<StyleData | null>(null);
  const [mode, setMode] = useState<'select' | 'create' | 'test' | 'apply'>('select');
  const [error, setError] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  const handleStyleCreate = async (file: File, name?: string) => {
    try {
      setError(null);
      setIsCreating(true);
      const style = await apiService.createStyle(file, { name });
      setSelectedStyle(style);
      // Don't auto-switch to live camera - user will click "Live Camera" button
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create style');
    } finally {
      setIsCreating(false);
    }
  };

  const handleStyleSelect = (style: StyleData) => {
    setSelectedStyle(style);
    setMode('test'); // Default to test mode first
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Real-Time Makeup Filter
          </h1>
          <p className="text-xl text-gray-600">
            Apply makeup styles in real-time using your camera
          </p>
        </div>

        {error && (
          <div className="max-w-4xl mx-auto mb-4 bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        <div className="max-w-6xl mx-auto">
          {mode === 'select' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex justify-end mb-4">
                  <button
                    onClick={() => setMode('create')}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                  >
                    Create New Style
                  </button>
                </div>

                <StyleSelector
                  onStyleSelect={handleStyleSelect}
                  selectedStyleId={selectedStyle?.style_id}
                />
              </div>
            </div>
          )}

          {mode === 'create' && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-2xl font-semibold mb-4">Create New Makeup Style</h2>
                <p className="text-gray-600 mb-4">
                  Upload an image with makeup to extract the style. The system will generate
                  LUTs and shaders for real-time application.
                </p>
                <StyleUpload 
                  onUpload={handleStyleCreate} 
                  isUploading={isCreating}
                  onStyleCreated={(style) => {
                    // Style created, but don't auto-switch mode yet
                    // User will click "Live Camera" button to switch
                    setSelectedStyle(style);
                  }}
                />
                <div className="flex gap-2 mt-4">
                  <button
                    onClick={() => setMode('select')}
                    className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
                  >
                    Back to Browse
                  </button>
                  {selectedStyle && (
                    <button
                      onClick={() => setMode('apply')}
                      className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 flex items-center gap-2"
                    >
                      <span>🎥</span>
                      Live Camera
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {mode === 'test' && selectedStyle && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h2 className="text-2xl font-semibold">Test Filter: {selectedStyle.name || selectedStyle.style_id}</h2>
                    {selectedStyle.description && (
                      <p className="text-gray-600">{selectedStyle.description}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setMode('apply')}
                      className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                      Live Camera
                    </button>
                    <button
                      onClick={() => {
                        setMode('select');
                        setSelectedStyle(null);
                      }}
                      className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
                    >
                      Change Style
                    </button>
                  </div>
                </div>
              </div>

              <FilterTester style={selectedStyle} />
            </div>
          )}

          {mode === 'apply' && selectedStyle && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h2 className="text-2xl font-semibold">Live Camera: {selectedStyle.name || selectedStyle.style_id}</h2>
                    {selectedStyle.description && (
                      <p className="text-gray-600">{selectedStyle.description}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setMode('test')}
                      className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                      Test on Image
                    </button>
                    <button
                      onClick={() => {
                        setMode('select');
                        setSelectedStyle(null);
                      }}
                      className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
                    >
                      Change Style
                    </button>
                  </div>
                </div>
              </div>

              <CameraFilter
                style={selectedStyle}
                onError={(err) => setError(err.message)}
              />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

