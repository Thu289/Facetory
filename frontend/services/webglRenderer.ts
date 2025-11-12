import type { StyleData, RegionMesh } from './api';
import type { FaceLandmarks } from './mediapipe';

type RegionKey =
  | 'lips'
  | 'lips_upper'
  | 'lips_lower'
  | 'eyebrows'
  | 'eyebrow_left'
  | 'eyebrow_right'
  | 'skin';

type OverlayColor = [number, number, number];

type OverlayStatus = 'mesh' | 'average';

interface WarpResult {
  rgb: Float32Array;
  alpha: Float32Array;
}

interface Point {
  x: number;
  y: number;
}

interface SampleRGBA {
  r: number;
  g: number;
  b: number;
  a: number;
}

const MESH_REGION_FALLBACKS: Partial<Record<RegionKey, RegionKey>> = {
  lips_upper: 'lips',
  lips_lower: 'lips',
  eyebrow_left: 'eyebrows',
  eyebrow_right: 'eyebrows',
};

const SOFTLIGHT_REGIONS: Set<RegionKey> = new Set();

interface RenderState {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
}

class RegionRenderer {
  private state: RenderState | null = null;
  private currentStyle: StyleData | null = null;
  private overlayColorMap: Partial<Record<RegionKey, OverlayColor>> = {};
  private overlayCacheKey: string | null = null;
  private overlayImageMap: Partial<Record<RegionKey, ImageData>> = {};
  private overlayStatus: Partial<Record<RegionKey, OverlayStatus>> = {};
  private regionMeshes: Record<string, RegionMesh> = {};
  private landmarkWarningShown = false;

  async initialize(canvas: HTMLCanvasElement, style: StyleData): Promise<void> {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('2D canvas context not available');
    }

    this.state = { canvas, ctx };
    this.currentStyle = style;
    this.loadRegionMeshes(style);
    await this.loadOverlayAssets(style);
  }

  async updateStyle(style: StyleData): Promise<void> {
    this.currentStyle = style;
    this.loadRegionMeshes(style);
    await this.loadOverlayAssets(style);
  }

  createVideoTexture(video: HTMLVideoElement): HTMLVideoElement {
    return video;
  }

  updateVideoTexture(_video: HTMLVideoElement, _texture: HTMLVideoElement): void {
    // No-op kept for API compatibility
  }

  render(
    video: HTMLVideoElement,
    regionMasks: Record<string, ImageData>,
    width: number,
    height: number,
    faceMask?: ImageData,
    landmarks?: FaceLandmarks | null,
  ): void {
    if (!this.state) {
      return;
    }

    const { canvas, ctx } = this.state;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.drawImage(video, 0, 0, width, height);

    const frame = ctx.getImageData(0, 0, width, height);
    const data = frame.data;
    const faceData = faceMask ? faceMask.data : null;
    const hasLandmarks =
      Boolean(landmarks && Array.isArray(landmarks.landmarks) && landmarks.landmarks.length > 0);

    if (!hasLandmarks && Object.keys(this.regionMeshes).length > 0 && !this.landmarkWarningShown) {
      console.warn('[RegionRenderer] Face landmarks unavailable; using fallback blending.');
      this.landmarkWarningShown = true;
    } else if (hasLandmarks && this.landmarkWarningShown) {
      this.landmarkWarningShown = false;
    }

    const applyRegion = (region: RegionKey) => {
      const color = this.getRegionColor(region);
      if (!color) {
        return;
      }

      const coverage = this.getRegionCoverage(region);
      if (coverage <= 0) {
        return;
      }

      const mask = regionMasks[region];
      const maskData = mask ? mask.data : null;
      const overlay = this.overlayImageMap[region];
      const mesh = this.resolveRegionMesh(region);

      if (overlay && mesh && hasLandmarks && landmarks) {
        const warpResult = this.warpOverlayWithMesh(overlay, mesh, landmarks, width, height);
        if (warpResult) {
          if (this.overlayStatus[region] !== 'mesh') {
            console.info(`[RegionRenderer] Using mesh-warped overlay for region "${region}"`);
            this.overlayStatus[region] = 'mesh';
          }
          const warpRgb = warpResult.rgb;
          const warpAlpha = warpResult.alpha;
          const useSoftlight = SOFTLIGHT_REGIONS.has(region);
          for (let i = 0; i < data.length; i += 4) {
            const pixelIndex = i / 4;
            const overlayAlpha = warpAlpha[pixelIndex];
            if (overlayAlpha <= 0.001) {
              continue;
            }

            const maskAlpha = maskData ? maskData[i] / 255 : 1;
            if (maskData && maskAlpha <= 0.001) {
              continue;
            }

            const faceAlpha = faceData ? faceData[i] / 255 : 1.0;
            const alpha = overlayAlpha * maskAlpha * faceAlpha * coverage;
            if (alpha <= 0.001) {
              continue;
            }

            const rgbIndex = pixelIndex * 3;
            if (useSoftlight) {
              const softR = this.applySoftlight(data[i], warpRgb[rgbIndex]);
              const softG = this.applySoftlight(data[i + 1], warpRgb[rgbIndex + 1]);
              const softB = this.applySoftlight(data[i + 2], warpRgb[rgbIndex + 2]);
              data[i] = data[i] * (1 - alpha) + softR * alpha;
              data[i + 1] = data[i + 1] * (1 - alpha) + softG * alpha;
              data[i + 2] = data[i + 2] * (1 - alpha) + softB * alpha;
            } else {
              data[i] = data[i] * (1 - alpha) + warpRgb[rgbIndex] * alpha;
              data[i + 1] = data[i + 1] * (1 - alpha) + warpRgb[rgbIndex + 1] * alpha;
              data[i + 2] = data[i + 2] * (1 - alpha) + warpRgb[rgbIndex + 2] * alpha;
            }
          }
          return;
        }
      }

      if (maskData) {
        if (this.overlayStatus[region] !== 'average') {
          console.info(`[RegionRenderer] Fallback to average RGB for region "${region}"`);
          this.overlayStatus[region] = 'average';
        }
        for (let i = 0; i < data.length; i += 4) {
          const maskAlpha = maskData[i] / 255;
          if (maskAlpha <= 0.01) {
            continue;
          }

          const faceAlpha = faceData ? faceData[i] / 255 : 1.0;
          const alpha = maskAlpha * faceAlpha * coverage;
          if (alpha <= 0.001) {
            continue;
          }

          data[i] = data[i] * (1 - alpha) + color[0] * alpha;
          data[i + 1] = data[i + 1] * (1 - alpha) + color[1] * alpha;
          data[i + 2] = data[i + 2] * (1 - alpha) + color[2] * alpha;
        }
      }
    };

    const lipsRegions: RegionKey[] =
      regionMasks['lips_upper'] || regionMasks['lips_lower']
        ? (['lips_upper', 'lips_lower'] as RegionKey[])
        : (['lips'] as RegionKey[]);

    const browRegions: RegionKey[] =
      regionMasks['eyebrow_left'] || regionMasks['eyebrow_right']
        ? (['eyebrow_left', 'eyebrow_right'] as RegionKey[])
        : (['eyebrows'] as RegionKey[]);

    const regions: RegionKey[] = ['skin', ...lipsRegions, ...browRegions];
    regions.forEach((region) => applyRegion(region));

    ctx.putImageData(frame, 0, 0);
  }

  cleanup(): void {
    this.state = null;
    this.currentStyle = null;
  }

  private getRegionCoverage(region: RegionKey): number {
    const params = this.getRegionParams(region);
    const coverage = typeof params?.coverage_intensity === 'number' ? params.coverage_intensity : 1.0;
    return Math.max(0, Math.min(1, coverage));
  }

  private getRegionColor(region: RegionKey): [number, number, number] | null {
    const overlayColor = this.overlayColorMap[region];
    if (overlayColor) {
      return overlayColor;
    }

    const params = this.getRegionParams(region);
    if (Array.isArray(params?.average_rgb) && params.average_rgb.length === 3) {
      return [
        this.clampColor(params.average_rgb[0]),
        this.clampColor(params.average_rgb[1]),
        this.clampColor(params.average_rgb[2]),
      ];
    }

    if ((region === 'lips_upper' || region === 'lips_lower') && this.getRegionParams('lips')?.average_rgb) {
      const base = this.getRegionParams('lips')!.average_rgb!;
      return [this.clampColor(base[0]), this.clampColor(base[1]), this.clampColor(base[2])];
    }

    return null;
  }

  private getRegionParams(region: RegionKey) {
    if (!this.currentStyle) {
      return undefined;
    }

    const params = this.currentStyle.style_parameters ?? {};
    if (region in params) {
      return params[region as keyof typeof params] as Record<string, any>;
    }

    const topLevel = (this.currentStyle as Record<string, any>)[region];
    if (topLevel && typeof topLevel === 'object') {
      return topLevel;
    }

    return undefined;
  }

  private clampColor(value: number): number {
    if (Number.isNaN(value)) {
      return 0;
    }
    return Math.max(0, Math.min(255, Math.round(value)));
  }

  private async loadOverlayAssets(style: StyleData): Promise<void> {
    const regionMasks = style.download_urls?.region_masks;
    if (!regionMasks) {
      this.overlayColorMap = {};
      this.overlayImageMap = {};
      this.overlayCacheKey = null;
      return;
    }

    const cacheKey = this.createOverlayCacheKey(style.style_id, regionMasks);
    if (this.overlayCacheKey === cacheKey) {
      return;
    }

    this.overlayCacheKey = cacheKey;
    this.overlayColorMap = {};
    this.overlayImageMap = {};
    this.overlayStatus = {};

    const supportedRegions: RegionKey[] = [
      'lips',
      'lips_upper',
      'lips_lower',
      'eyebrows',
      'eyebrow_left',
      'eyebrow_right',
      'skin',
    ];
    await Promise.all(
      supportedRegions
        .map(async (region) => {
          const url = regionMasks[region];
          if (!url) {
            return;
          }

          const imageData = await this.fetchImageData(url);
          if (!imageData) {
            return;
          }

          this.overlayImageMap[region] = imageData;
          const color = this.computeAverageColor(imageData);
          if (color) {
            this.overlayColorMap[region] = color;
          }
        }),
    );
  }

  private loadRegionMeshes(style: StyleData): void {
    const meshes =
      style.metadata?.region_meshes ??
      (style as unknown as { region_meshes?: Record<string, RegionMesh> }).region_meshes ??
      {};
    this.regionMeshes = meshes ? { ...meshes } : {};
    this.landmarkWarningShown = false;
  }

  private resolveRegionMesh(region: RegionKey): RegionMesh | undefined {
    if (this.regionMeshes[region]) {
      return this.regionMeshes[region];
    }
    const fallback = MESH_REGION_FALLBACKS[region];
    if (fallback && this.regionMeshes[fallback]) {
      return this.regionMeshes[fallback];
    }
    return undefined;
  }

  private sampleOverlay(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    x: number,
    y: number,
  ): SampleRGBA {
    const clampedX = Math.max(0, Math.min(width - 1, x));
    const clampedY = Math.max(0, Math.min(height - 1, y));

    const x0 = Math.floor(clampedX);
    const y0 = Math.floor(clampedY);
    const x1 = Math.min(x0 + 1, width - 1);
    const y1 = Math.min(y0 + 1, height - 1);

    const tx = clampedX - x0;
    const ty = clampedY - y0;

    const idx = (ix: number, iy: number) => (iy * width + ix) * 4;

    const i00 = idx(x0, y0);
    const i10 = idx(x1, y0);
    const i01 = idx(x0, y1);
    const i11 = idx(x1, y1);

    const w00 = (1 - tx) * (1 - ty);
    const w10 = tx * (1 - ty);
    const w01 = (1 - tx) * ty;
    const w11 = tx * ty;

    const r =
      data[i00] * w00 + data[i10] * w10 + data[i01] * w01 + data[i11] * w11;
    const g =
      data[i00 + 1] * w00 + data[i10 + 1] * w10 + data[i01 + 1] * w01 + data[i11 + 1] * w11;
    const b =
      data[i00 + 2] * w00 + data[i10 + 2] * w10 + data[i01 + 2] * w01 + data[i11 + 2] * w11;
    const a =
      data[i00 + 3] * w00 + data[i10 + 3] * w10 + data[i01 + 3] * w01 + data[i11 + 3] * w11;

    return { r, g, b, a: a / 255 };
  }

  private computeBarycentric(triangle: [Point, Point, Point], px: number, py: number): [number, number, number] | null {
    const [p0, p1, p2] = triangle;
    const denom =
      (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y);
    if (Math.abs(denom) < 1e-6) {
      return null;
    }
    const w0 =
      ((p1.y - p2.y) * (px - p2.x) + (p2.x - p1.x) * (py - p2.y)) / denom;
    const w1 =
      ((p2.y - p0.y) * (px - p2.x) + (p0.x - p2.x) * (py - p2.y)) / denom;
    const w2 = 1 - w0 - w1;
    return [w0, w1, w2];
  }

  private warpOverlayWithMesh(
    overlay: ImageData,
    mesh: RegionMesh,
    landmarks: FaceLandmarks,
    targetWidth: number,
    targetHeight: number,
  ): WarpResult | null {
    if (!mesh.points || !mesh.triangles || mesh.points.length < 3) {
      return null;
    }

    const overlayWidth = overlay.width;
    const overlayHeight = overlay.height;
    const overlayData = overlay.data;

    const srcPoints: Point[] = [];
    const dstPoints: Point[] = [];

    for (const point of mesh.points) {
      const lm = landmarks.landmarks[point.index];
      if (!lm) {
        return null;
      }
      srcPoints.push({
        x: point.x * overlayWidth,
        y: point.y * overlayHeight,
      });
      dstPoints.push({
        x: lm.x * targetWidth,
        y: lm.y * targetHeight,
      });
    }

    const rgb = new Float32Array(targetWidth * targetHeight * 3);
    const alpha = new Float32Array(targetWidth * targetHeight);

    for (const tri of mesh.triangles) {
      if (!Array.isArray(tri) || tri.length !== 3) {
        continue;
      }
      const [i0, i1, i2] = tri as [number, number, number];
      const srcTri: [Point, Point, Point] = [srcPoints[i0], srcPoints[i1], srcPoints[i2]];
      const dstTri: [Point, Point, Point] = [dstPoints[i0], dstPoints[i1], dstPoints[i2]];

      const area =
        (dstTri[1].x - dstTri[0].x) * (dstTri[2].y - dstTri[0].y) -
        (dstTri[2].x - dstTri[0].x) * (dstTri[1].y - dstTri[0].y);
      if (Math.abs(area) < 1e-3) {
        continue;
      }

      const minX = Math.max(
        0,
        Math.floor(Math.min(dstTri[0].x, dstTri[1].x, dstTri[2].x)),
      );
      const maxX = Math.min(
        targetWidth - 1,
        Math.ceil(Math.max(dstTri[0].x, dstTri[1].x, dstTri[2].x)),
      );
      const minY = Math.max(
        0,
        Math.floor(Math.min(dstTri[0].y, dstTri[1].y, dstTri[2].y)),
      );
      const maxY = Math.min(
        targetHeight - 1,
        Math.ceil(Math.max(dstTri[0].y, dstTri[1].y, dstTri[2].y)),
      );

      for (let y = minY; y <= maxY; y += 1) {
        for (let x = minX; x <= maxX; x += 1) {
          const bary = this.computeBarycentric(dstTri, x + 0.5, y + 0.5);
          if (!bary) {
            continue;
          }
          const [w0, w1, w2] = bary;
          if (w0 < -0.01 || w1 < -0.01 || w2 < -0.01) {
            continue;
          }

          const srcX = w0 * srcTri[0].x + w1 * srcTri[1].x + w2 * srcTri[2].x;
          const srcY = w0 * srcTri[0].y + w1 * srcTri[1].y + w2 * srcTri[2].y;
          const sample = this.sampleOverlay(overlayData, overlayWidth, overlayHeight, srcX, srcY);
          if (sample.a <= 0.01) {
            continue;
          }

          const pixelIndex = y * targetWidth + x;
          const rgbIndex = pixelIndex * 3;
          const prevAlpha = alpha[pixelIndex];
          const newAlpha = sample.a + prevAlpha * (1 - sample.a);
          if (newAlpha <= 1e-6) {
            continue;
          }

          rgb[rgbIndex] =
            (rgb[rgbIndex] * prevAlpha + sample.r * sample.a * (1 - prevAlpha)) /
            newAlpha;
          rgb[rgbIndex + 1] =
            (rgb[rgbIndex + 1] * prevAlpha + sample.g * sample.a * (1 - prevAlpha)) /
            newAlpha;
          rgb[rgbIndex + 2] =
            (rgb[rgbIndex + 2] * prevAlpha + sample.b * sample.a * (1 - prevAlpha)) /
            newAlpha;
          alpha[pixelIndex] = newAlpha;
        }
      }
    }

    return { rgb, alpha };
  }

  private applySoftlight(base: number, overlay: number): number {
    const baseNorm = Math.max(0, Math.min(1, base / 255));
    const overlayNorm = Math.max(0, Math.min(1, overlay / 255));
    let out: number;
    if (overlayNorm <= 0.5) {
      out = baseNorm - (1 - 2 * overlayNorm) * baseNorm * (1 - baseNorm);
    } else {
      out = baseNorm + (2 * overlayNorm - 1) * (Math.sqrt(baseNorm) - baseNorm);
    }
    return Math.max(0, Math.min(255, out * 255));
  }

  private async fetchImageData(url: string): Promise<ImageData | null> {
    try {
      const response = await fetch(url, { mode: 'cors' });
      if (!response.ok) {
        return null;
      }

      const blob = await response.blob();
      const image = await this.loadImageFromBlob(blob);
      if (!image) {
        return null;
      }

      const width = image.naturalWidth || image.width;
      const height = image.naturalHeight || image.height;
      if (!width || !height) {
        return null;
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return null;
      }

      ctx.drawImage(image, 0, 0, width, height);
      const imageData = ctx.getImageData(0, 0, width, height);
      return imageData;
    } catch (error) {
      console.warn('[RegionRenderer] Failed to fetch region overlay', url, error);
      return null;
    }
  }

  private loadImageFromBlob(blob: Blob): Promise<HTMLImageElement | null> {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(null);
      };
      img.src = url;
    });
  }

  private computeAverageColor(imageData: ImageData): OverlayColor | null {
    const data = imageData.data;
    let r = 0;
    let g = 0;
    let b = 0;
    let count = 0;

    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3];
      if (alpha < 4) {
        continue;
      }
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      count += 1;
    }

    if (count === 0) {
      return null;
    }

    return [r / count, g / count, b / count].map((value) => Math.round(value)) as OverlayColor;
  }

  private createOverlayCacheKey(styleId: string, maskUrls: Record<string, string>): string {
    const parts = Object.entries(maskUrls)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([region, url]) => `${region}:${url}`);
    return `${styleId}|${parts.join('|')}`;
  }
}

export const webglRenderer = new RegionRenderer();

