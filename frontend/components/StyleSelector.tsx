'use client';

import React, { useEffect, useState } from 'react';
import { apiService, type StyleData } from '../services/api';
import RegionPreview from './RegionPreview';

interface StyleSelectorProps {
  onStyleSelect: (style: StyleData) => void;
  selectedStyleId?: string;
}

export default function StyleSelector({ onStyleSelect, selectedStyleId }: StyleSelectorProps) {
  const [styles, setStyles] = useState<StyleData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creatingDefault, setCreatingDefault] = useState(false);

  useEffect(() => {
    void loadStyles();
  }, []);

  const loadStyles = async () => {
    try {
      setLoading(true);
      const result = await apiService.listStyles();
      setStyles(result.styles);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load styles');
      console.error('Failed to load styles:', err);
    } finally {
      setLoading(false);
    }
  };

  const createDefaultStyle = async () => {
    try {
      setCreatingDefault(true);
      const defaultStyle = await apiService.createDefaultStyle();
      await loadStyles();
      onStyleSelect(defaultStyle);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create default style';
      setError(message);
      console.error('Failed to create default style:', err);
      alert(`Failed to create default style: ${message}`);
    } finally {
      setCreatingDefault(false);
    }
  };

  if (loading) {
    return (
      <div className="py-8 text-center">
        <p className="text-gray-600">Loading styles...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4">
        <p className="text-red-800">{error}</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg bg-white p-6 shadow">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-semibold">Select Makeup Style</h2>
        <button
          onClick={createDefaultStyle}
          disabled={creatingDefault}
          className="rounded bg-red-600 px-4 py-2 text-sm text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {creatingDefault ? 'Creating...' : 'Create Default Red Lips Filter'}
        </button>
      </div>

      {styles.length === 0 ? (
        <div className="py-8 text-center">
          <p className="mb-4 text-gray-600">No styles available yet.</p>
          <p className="mb-4 text-sm text-gray-500">
            Create your first style by uploading an image with makeup!
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {styles.map((style) => {
            const isSelected = selectedStyleId === style.style_id;
            return (
              <button
                type="button"
                key={style.style_id}
                onClick={() => onStyleSelect(style)}
                className={`flex flex-col gap-3 rounded-lg border p-4 text-left transition-all ${
                  isSelected
                    ? 'border-purple-600 bg-purple-50 shadow-md'
                    : 'border-gray-200 hover:border-purple-300'
                }`}
              >
                {style.metadata?.filter_preview && (
                  <img
                    src={style.metadata.filter_preview}
                    alt={style.name || style.style_id}
                    className="h-32 w-full rounded bg-gray-100 object-contain"
                  />
                )}

                <div>
                  <h3 className="font-semibold">{style.name || style.style_id}</h3>
                  {style.description && (
                    <p className="mt-1 text-sm text-gray-600">{style.description}</p>
                  )}
                  <p className="mt-1 text-xs text-gray-500">ID: {style.style_id}</p>
                </div>

                <RegionPreview style={style} />
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
