import React from 'react';
import type { RegionName, RegionStyle, StyleData } from '../services/api';

const REGION_LABELS: Record<RegionName, string> = {
  lips: 'Lips',
  lips_upper: 'Upper Lip',
  lips_lower: 'Lower Lip',
  eyes: 'Eyes',
  eyebrows: 'Eyebrows',
  eyebrow_left: 'Left Eyebrow',
  eyebrow_right: 'Right Eyebrow',
  skin: 'Skin',
  cheeks: 'Cheeks',
};

const REGION_ORDER: RegionName[] = [
  'lips',
  'lips_upper',
  'lips_lower',
  'eyebrows',
  'eyebrow_left',
  'eyebrow_right',
  'eyes',
  'skin',
  'cheeks',
];

interface RegionPreviewProps {
  style: StyleData;
  showMaskImages?: boolean;
  maxPreviewWidth?: number;
}

const rgbToHex = (rgb: [number, number, number]) =>
  `#${rgb
    .map((channel) => Math.max(0, Math.min(255, Math.round(channel))).toString(16).padStart(2, '0'))
    .join('')}`;

export function RegionPreview({
  style,
  showMaskImages = true,
  maxPreviewWidth = 96,
}: RegionPreviewProps): JSX.Element | null {
  const { style_parameters: styleParams, download_urls: downloadUrls, mask_previews } = style;

  if (!styleParams) {
    return null;
  }

  const items = REGION_ORDER.map((region) => {
    const regionData = styleParams[region] as RegionStyle | undefined;
    const average = regionData?.average_rgb;
    const maskUrl =
      downloadUrls?.region_masks?.[region] ?? mask_previews?.[region] ?? undefined;

    if (!average && !maskUrl) {
      return null;
    }

    const hex = average ? rgbToHex(average) : undefined;

    return (
      <div key={region} className="flex items-center gap-4 rounded-lg border border-gray-200 p-3">
        {average && (
          <div className="flex flex-col items-center gap-1">
            <div
              className="h-12 w-12 rounded-full border border-gray-300 shadow-inner"
              style={{ backgroundColor: `rgb(${average.join(',')})` }}
              title={hex}
            />
            <span className="text-xs text-gray-500">{hex}</span>
          </div>
        )}

        <div className="flex flex-1 flex-col gap-1">
          <span className="font-medium text-gray-800">{REGION_LABELS[region]}</span>
          {average && (
            <span className="text-xs text-gray-500">
              RGB({average.map((value) => Math.round(value)).join(', ')})
            </span>
          )}
        </div>

        {showMaskImages && maskUrl && (
          <div className="flex items-center justify-center">
            <img
              src={maskUrl}
              alt={`${REGION_LABELS[region]} overlay`}
              className="rounded border border-gray-200"
              style={{ maxWidth: maxPreviewWidth, maxHeight: maxPreviewWidth }}
            />
          </div>
        )}
      </div>
    );
  }).filter(Boolean);

  if (items.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">Region Colors & Masks</h4>
      <div className="grid gap-3">{items}</div>
    </div>
  );
}

export default RegionPreview;


