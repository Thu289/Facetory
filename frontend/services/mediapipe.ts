/**
 * MediaPipe FaceMesh Service
 * Handles face detection and landmark extraction
 */

// Dynamic imports to avoid SSR issues
// These will be loaded on-demand when needed
let FaceMesh: any;
let Camera: any;

export interface FaceLandmarks {
  landmarks: Array<{ x: number; y: number; z?: number }>;
  imageWidth: number;
  imageHeight: number;
}

export interface FaceRegion {
  lips: Array<{ x: number; y: number }>;
  leftEye: Array<{ x: number; y: number }>;
  rightEye: Array<{ x: number; y: number }>;
  leftEyebrow: Array<{ x: number; y: number }>;
  rightEyebrow: Array<{ x: number; y: number }>;
  faceOval: Array<{ x: number; y: number }>;
  nose: Array<{ x: number; y: number }>;
}

class MediaPipeService {
  private faceMesh: FaceMesh | null = null;
  private camera: Camera | null = null;
  private onResultsCallback: ((landmarks: FaceLandmarks) => void) | null = null;
  private frameCount: number = 0;

  /**
   * Initialize MediaPipe FaceMesh
   */
  async initialize(): Promise<void> {
    if (typeof window === 'undefined') {
      throw new Error('MediaPipe can only be used on client-side');
    }

    if (this.faceMesh) {
      console.log('ℹ️ [MediaPipe] FaceMesh already initialized');
      return; // Already initialized
    }

    console.log('🚀 [MediaPipe] Initializing FaceMesh...');
    // Dynamic import if not already loaded
    if (!FaceMesh) {
      const faceMeshModule = await import('@mediapipe/face_mesh');
      FaceMesh = faceMeshModule.FaceMesh;
    }

    this.faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      },
    });

    this.faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    console.log('✅ [MediaPipe] FaceMesh initialized with options:', {
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    this.faceMesh.onResults((results) => {
      this.frameCount++;
      
      const hasFace = results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0;
      
      if (this.onResultsCallback && hasFace) {
        const landmarks = results.multiFaceLandmarks[0];
        const imageWidth = results.image?.width || 640;
        const imageHeight = results.image?.height || 480;

        this.onResultsCallback({
          landmarks: landmarks.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
          imageWidth,
          imageHeight,
        });
      }
    });
  }

  /**
   * Start camera and face tracking
   */
  async startCamera(
    videoElement: HTMLVideoElement,
    onResults: (landmarks: FaceLandmarks) => void
  ): Promise<void> {
    if (typeof window === 'undefined') {
      throw new Error('Camera can only be used on client-side');
    }

    if (!this.faceMesh) {
      await this.initialize();
    }

    // Dynamic import if not already loaded
    if (!Camera) {
      const cameraModule = await import('@mediapipe/camera_utils');
      Camera = cameraModule.Camera;
    }

    this.onResultsCallback = onResults;

    if (this.camera) {
      this.camera.stop();
    }

    
    // Initialize frame counters
    (this as any).frameSendCount = 0;
    (this as any).frameSkipCount = 0;
    (this as any).animationFrameId = null;
    (this as any).videoElement = videoElement;
    
    // Instead of using MediaPipe Camera (which manages its own video element),
    // manually request animation frames and send frames to FaceMesh
    console.log('✅ [MediaPipe] Face tracking active');
    
    const processFrame = async () => {
      if (!videoElement || !this.faceMesh) {
        return;
      }
      
      const frameSendCount = (this as any).frameSendCount || 0;
      (this as any).frameSendCount = frameSendCount + 1;
      
      if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        await this.faceMesh!.send({ image: videoElement });
      }
      
      // Request next frame
      if ((this as any).animationFrameId !== null) {
        (this as any).animationFrameId = requestAnimationFrame(processFrame);
      }
    };
    
    // Start animation frame loop
    (this as any).animationFrameId = requestAnimationFrame(processFrame);
    
  }

  /**
   * Stop camera
   */
  stopCamera(): void {
    // Stop animation frame loop
    if ((this as any).animationFrameId !== null) {
      cancelAnimationFrame((this as any).animationFrameId);
      (this as any).animationFrameId = null;
    }
    
    // Also stop MediaPipe Camera if it exists
    if (this.camera) {
      this.camera.stop();
      this.camera = null;
    }
    this.onResultsCallback = null;
  }

  /**
   * Map face landmarks to makeup regions
   */
  mapLandmarksToRegions(landmarks: FaceLandmarks): FaceRegion {
    // MediaPipe FaceMesh landmark indices
    // Based on https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
    const landmarkIndices = {
      lips: [
        61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
        13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
      ],
      leftEye: [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
      ],
      rightEye: [
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
      ],
      leftEyebrow: [107, 55, 65, 52, 53, 46],
      rightEyebrow: [336, 296, 334, 293, 300, 276],
      faceOval: [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
      ],
      nose: [4, 51, 48, 115, 131, 134, 102, 49, 220, 305, 290, 305],
    };

    const mappedRegions: FaceRegion = {
      lips: [],
      leftEye: [],
      rightEye: [],
      leftEyebrow: [],
      rightEyebrow: [],
      faceOval: [],
      nose: [],
    };

    Object.entries(landmarkIndices).forEach(([region, indices]) => {
      const points = indices
        .map((idx) => {
          const lm = landmarks.landmarks[idx];
          if (!lm) return null;
          return { x: lm.x * landmarks.imageWidth, y: lm.y * landmarks.imageHeight };
        })
        .filter((p): p is { x: number; y: number } => p !== null);
      
      mappedRegions[region as keyof FaceRegion] = points;
    });

    return mappedRegions;
  }

  /**
   * Generate region masks from landmarks
   * Returns masks with standardized names: lips, eyes, eyebrows, skin, cheeks, face
   */
  generateRegionMasks(
    landmarks: FaceLandmarks,
    width: number,
    height: number
  ): Record<string, ImageData> {
    // Throttle logs - only log every 30 calls
    if (this.frameCount % 30 === 0) {
      console.log(`🎭 [MediaPipe] Generating region masks from ${landmarks.landmarks.length} landmarks`);
    }
    const regions = this.mapLandmarksToRegions(landmarks);
    const masks: Record<string, ImageData> = {};

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;

    // Helper to create mask from points
    const createMaskFromPoints = (points: Array<{ x: number; y: number }>): ImageData | null => {
      if (points.length === 0) return null;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'white';
      ctx.beginPath();

      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.closePath();
      ctx.fill();

      return ctx.getImageData(0, 0, width, height);
    };

    // Lips mask
    let lipsMask: ImageData | null = null;
    if (regions.lips.length > 0) {
      lipsMask = createMaskFromPoints(regions.lips);
      if (lipsMask) masks['lips'] = lipsMask;
    }

    if (lipsMask) {
      const upperIndex = 13;
      const lowerIndex = 14;
      const upperLandmarkY = landmarks.landmarks[upperIndex]?.y ?? 0.5;
      const lowerLandmarkY = landmarks.landmarks[lowerIndex]?.y ?? 0.6;
      const splitLine = ((upperLandmarkY * height) + (lowerLandmarkY * height)) / 2;

      const upperLipData = new Uint8ClampedArray(lipsMask.data);
      const lowerLipData = new Uint8ClampedArray(lipsMask.data);

      for (let i = 0; i < lipsMask.data.length; i += 4) {
        const pixelIndex = i / 4;
        const y = Math.floor(pixelIndex / width);

        if (y >= splitLine) {
          // Below split line -> remove from upper lip
          upperLipData[i] = 0;
          upperLipData[i + 1] = 0;
          upperLipData[i + 2] = 0;
          upperLipData[i + 3] = 0;
        } else {
          // Above split line -> remove from lower lip
          lowerLipData[i] = 0;
          lowerLipData[i + 1] = 0;
          lowerLipData[i + 2] = 0;
          lowerLipData[i + 3] = 0;
        }
      }

      masks['lips_upper'] = new ImageData(upperLipData, width, height);
      masks['lips_lower'] = new ImageData(lowerLipData, width, height);
    }

    // Eyes mask: combine leftEye and rightEye
    if (regions.leftEye.length > 0 || regions.rightEye.length > 0) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'white';
      // Left eye
      if (regions.leftEye.length > 0) {
        ctx.beginPath();
        ctx.moveTo(regions.leftEye[0].x, regions.leftEye[0].y);
        for (let i = 1; i < regions.leftEye.length; i++) {
          ctx.lineTo(regions.leftEye[i].x, regions.leftEye[i].y);
        }
        ctx.closePath();
        ctx.fill();
      }
      // Right eye
      if (regions.rightEye.length > 0) {
        ctx.beginPath();
        ctx.moveTo(regions.rightEye[0].x, regions.rightEye[0].y);
        for (let i = 1; i < regions.rightEye.length; i++) {
          ctx.lineTo(regions.rightEye[i].x, regions.rightEye[i].y);
        }
        ctx.closePath();
        ctx.fill();
      }
      masks['eyes'] = ctx.getImageData(0, 0, width, height);
    }

    // Eyebrows mask: separate left/right and combined
    if (regions.leftEyebrow.length > 0) {
      const mask = createMaskFromPoints(regions.leftEyebrow);
      if (mask) {
        masks['eyebrow_left'] = mask;
      }
    }
    if (regions.rightEyebrow.length > 0) {
      const mask = createMaskFromPoints(regions.rightEyebrow);
      if (mask) {
        masks['eyebrow_right'] = mask;
      }
    }
    if (regions.leftEyebrow.length > 0 || regions.rightEyebrow.length > 0) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'white';
      if (regions.leftEyebrow.length > 0) {
        ctx.beginPath();
        ctx.moveTo(regions.leftEyebrow[0].x, regions.leftEyebrow[0].y);
        for (let i = 1; i < regions.leftEyebrow.length; i++) {
          ctx.lineTo(regions.leftEyebrow[i].x, regions.leftEyebrow[i].y);
        }
        ctx.closePath();
        ctx.fill();
      }
      if (regions.rightEyebrow.length > 0) {
        ctx.beginPath();
        ctx.moveTo(regions.rightEyebrow[0].x, regions.rightEyebrow[0].y);
        for (let i = 1; i < regions.rightEyebrow.length; i++) {
          ctx.lineTo(regions.rightEyebrow[i].x, regions.rightEyebrow[i].y);
        }
        ctx.closePath();
        ctx.fill();
      }
      masks['eyebrows'] = ctx.getImageData(0, 0, width, height);
    }

    // Nose mask
    if (regions.nose.length > 0) {
      const noseMask = createMaskFromPoints(regions.nose);
      if (noseMask) masks['nose'] = noseMask;
    }

    // Skin mask: use faceOval as approximation for skin region
    if (regions.faceOval.length > 0) {
      const faceMask = createMaskFromPoints(regions.faceOval);
      if (faceMask) {
        // Subtract eyes, eyebrows, lips, nose from face to get skin
        const skinMaskData = new ImageData(new Uint8ClampedArray(faceMask.data), width, height);
        
        // Subtract other regions
        if (masks['eyes']) {
          for (let i = 0; i < skinMaskData.data.length; i += 4) {
            if (masks['eyes'].data[i] > 128) {
              skinMaskData.data[i] = 0;
              skinMaskData.data[i + 1] = 0;
              skinMaskData.data[i + 2] = 0;
              skinMaskData.data[i + 3] = 255;
            }
          }
        }
        if (masks['eyebrows']) {
          for (let i = 0; i < skinMaskData.data.length; i += 4) {
            if (masks['eyebrows'].data[i] > 128) {
              skinMaskData.data[i] = 0;
              skinMaskData.data[i + 1] = 0;
              skinMaskData.data[i + 2] = 0;
              skinMaskData.data[i + 3] = 255;
            }
          }
        }
        if (masks['eyebrow_left']) {
          for (let i = 0; i < skinMaskData.data.length; i += 4) {
            if (masks['eyebrow_left'].data[i] > 128) {
              skinMaskData.data[i] = 0;
              skinMaskData.data[i + 1] = 0;
              skinMaskData.data[i + 2] = 0;
              skinMaskData.data[i + 3] = 255;
            }
          }
        }
        if (masks['eyebrow_right']) {
          for (let i = 0; i < skinMaskData.data.length; i += 4) {
            if (masks['eyebrow_right'].data[i] > 128) {
              skinMaskData.data[i] = 0;
              skinMaskData.data[i + 1] = 0;
              skinMaskData.data[i + 2] = 0;
              skinMaskData.data[i + 3] = 255;
            }
          }
        }
        if (masks['lips']) {
          for (let i = 0; i < skinMaskData.data.length; i += 4) {
            if (masks['lips'].data[i] > 128) {
              skinMaskData.data[i] = 0;
              skinMaskData.data[i + 1] = 0;
              skinMaskData.data[i + 2] = 0;
              skinMaskData.data[i + 3] = 255;
            }
          }
        }
        if (masks['lips_upper']) {
          for (let i = 0; i < skinMaskData.data.length; i += 4) {
            if (masks['lips_upper'].data[i] > 128) {
              skinMaskData.data[i] = 0;
              skinMaskData.data[i + 1] = 0;
              skinMaskData.data[i + 2] = 0;
              skinMaskData.data[i + 3] = 255;
            }
          }
        }
        if (masks['lips_lower']) {
          for (let i = 0; i < skinMaskData.data.length; i += 4) {
            if (masks['lips_lower'].data[i] > 128) {
              skinMaskData.data[i] = 0;
              skinMaskData.data[i + 1] = 0;
              skinMaskData.data[i + 2] = 0;
              skinMaskData.data[i + 3] = 255;
            }
          }
        }
        if (masks['nose']) {
          for (let i = 0; i < skinMaskData.data.length; i += 4) {
            if (masks['nose'].data[i] > 128) {
              skinMaskData.data[i] = 0;
              skinMaskData.data[i + 1] = 0;
              skinMaskData.data[i + 2] = 0;
              skinMaskData.data[i + 3] = 255;
            }
          }
        }
        masks['skin'] = skinMaskData;
      }
    }

    // Face mask: use faceOval for overall face region
    if (regions.faceOval.length > 0) {
      const faceMask = createMaskFromPoints(regions.faceOval);
      if (faceMask) {
        masks['faceOval'] = faceMask;
        masks['face'] = faceMask; // Also provide as 'face' for compatibility

        // Ensure we always have a skin mask for downstream shaders
        if (!masks['skin']) {
          const faceCopy = new ImageData(new Uint8ClampedArray(faceMask.data), width, height);
          masks['skin'] = faceCopy;
        }
      }
    }
    

    return masks;
  }
}

export const mediaPipeService = new MediaPipeService();

