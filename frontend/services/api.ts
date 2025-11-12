/**
 * API Service for communicating with backend
 */

import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export type RegionName =
  | 'lips'
  | 'lips_upper'
  | 'lips_lower'
  | 'eyes'
  | 'eyebrows'
  | 'eyebrow_left'
  | 'eyebrow_right'
  | 'skin'
  | 'cheeks';

export interface RegionStyle {
  average_rgb?: [number, number, number];
  coverage_intensity?: number;
  [key: string]: unknown;
}

export interface RegionMeshPoint {
  index: number;
  x: number;
  y: number;
}

export interface RegionMesh {
  image_size: [number, number];
  points: RegionMeshPoint[];
  triangles: [number, number, number][];
}

export interface StyleData {
  style_id: string;
  name?: string;
  description?: string;
  download_urls: {
    region_masks?: Record<string, string>;
    style_parameters: string;
  };
  style_parameters: Partial<Record<RegionName, RegionStyle>>;
  storage_info: {
    minio_bucket: string;
    storage_path: string;
    database_stored: boolean;
  };
  metadata: {
    filter_preview?: string;
    original_cropped?: string;
    segmentation: any;
    face_detection: any;
    region_meshes?: Record<string, RegionMesh>;
  };
  region_meshes?: Record<string, RegionMesh>;
  mask_previews?: Record<string, string>;
  regions_detected?: string[];
}

class ApiService {
  private client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
  });

  /**
   * Create a complete style from an image
   */
  async createStyle(file: File, options?: { name?: string; description?: string; previewFile?: File }): Promise<StyleData> {
    const formData = new FormData();
    formData.append('file', file);
    if (options?.previewFile) {
      formData.append('preview_file', options.previewFile);
    }
    if (options?.name) formData.append('name', options.name);
    if (options?.description) formData.append('description', options.description);
    formData.append('store_in_db', 'false');

    const response = await this.client.post<StyleData>('/api/makeup/style/create_complete', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  /**
   * Get style information by ID
   */
  async getStyle(styleId: string): Promise<StyleData> {
    const response = await this.client.get<StyleData>(`/api/makeup/style/${styleId}`);
    return response.data;
  }

  /**
   * List all available styles
   */
  async listStyles(limit = 20, offset = 0): Promise<{ styles: StyleData[]; total: number }> {
    const response = await this.client.get('/api/makeup/styles', {
      params: { limit, offset },
    });
    return response.data;
  }

  /**
   * Extract style from image (without generating LUTs/shaders)
   */
  async extractStyle(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post('/api/face/makeup/style_extract', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  /**
   * Create a default red lipstick filter for testing
   */
  async createDefaultStyle(): Promise<StyleData> {
    const response = await this.client.post<StyleData>('/api/makeup/style/create_default');
    return response.data;
  }
}

export const apiService = new ApiService();

