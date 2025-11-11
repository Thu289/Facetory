'use client';

import React from 'react';
import SegmentationPreview from '../../components/SegmentationPreview';

export default function SegmentationPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Face Segmentation Preview
          </h1>
          <p className="text-xl text-gray-600">
            Test BiSeNet face segmentation model and view results
          </p>
        </div>

        <div className="max-w-6xl mx-auto">
          <SegmentationPreview />
        </div>
      </div>
    </main>
  );
}

