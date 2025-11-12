'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { StyleData } from '../services/api';
import type { FaceLandmarks } from '../services/mediapipe';

interface CameraFilterProps {
  style: StyleData;
  onError?: (error: Error) => void;
}

export default function CameraFilter({ style, onError }: CameraFilterProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const [isInitialized, setIsInitialized] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRegionFilters, setShowRegionFilters] = useState(false);
  const [detectedRegions, setDetectedRegions] = useState<string[]>([]);
  const faceMaskRef = useRef<ImageData | null>(null);
  const regionMasksRef = useRef<Record<string, ImageData>>({});
  const landmarksRef = useRef<FaceLandmarks | null>(null);

  // Initialize camera and WebGL
  useEffect(() => {
    let mounted = true;

    const initialize = async () => {
      try {
        if (typeof window === 'undefined') {
          return; // Skip on server-side
        }

        if (!videoRef.current || !canvasRef.current) {
          return;
        }

        // Load services dynamically
        const { mediaPipeService } = await import('../services/mediapipe');
        const { webglRenderer: renderer } = await import('../services/webglRenderer');

        // Initialize MediaPipe
        await mediaPipeService.initialize();

        // Setup video stream
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });

        if (!mounted) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.autoplay = true;
          videoRef.current.playsInline = true;
          
          // Wait for video to be ready
          await new Promise((resolve, reject) => {
            if (!videoRef.current) {
              reject(new Error('Video element not found'));
              return;
            }
            
            const video = videoRef.current;
            let resolved = false;
            
            const onReady = () => {
              if (resolved) return;
              resolved = true;
                video.play().then(() => {
                resolve(true);
              }).catch((err) => {
                console.error('❌ [CameraFilter] Video play failed:', err);
                reject(err);
              });
            };
            
            if (video.readyState >= 2) {
              // Video already loaded
              onReady();
            } else {
              video.onloadedmetadata = onReady;
              video.oncanplay = onReady;
            }
            
            // Timeout after 5 seconds
            setTimeout(() => {
              if (!resolved) {
                resolved = true;
                reject(new Error('Video load timeout'));
              }
            }, 5000);
          });
          
          // Wait a bit more to ensure video is actually playing
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Initialize WebGL renderer
        await renderer.initialize(canvasRef.current, style);

        // Setup MediaPipe face tracking callback
        // Verify video is ready before starting MediaPipe
        if (!videoRef.current || videoRef.current.readyState < 2) {
          console.error('❌ [CameraFilter] Video not ready before starting MediaPipe');
          throw new Error('Video element not ready');
        }
        
        await mediaPipeService.startCamera(videoRef.current!, (landmarks) => {
          if (canvasRef.current && videoRef.current) {
            landmarksRef.current = landmarks;
            const generatedMasks = mediaPipeService.generateRegionMasks(
              landmarks,
              canvasRef.current.width,
              canvasRef.current.height
            );

            const filteredMasks: Record<string, ImageData> = {};

            ['skin'].forEach((region) => {
              const mask = generatedMasks[region];
              if (mask) {
                filteredMasks[region] = mask;
              }
            });

            const lipRegions = ['lips_upper', 'lips_lower'] as const;
            let lipDetected = false;
            lipRegions.forEach((region) => {
              const mask = generatedMasks[region];
              if (mask) {
                filteredMasks[region] = mask;
                lipDetected = true;
              }
            });
            if (!lipDetected && generatedMasks.lips) {
              filteredMasks.lips = generatedMasks.lips;
            }

            const browRegions = ['eyebrow_left', 'eyebrow_right'] as const;
            let browDetected = false;
            browRegions.forEach((region) => {
              const mask = generatedMasks[region];
              if (mask) {
                filteredMasks[region] = mask;
                browDetected = true;
              }
            });
            if (!browDetected && generatedMasks.eyebrows) {
              filteredMasks.eyebrows = generatedMasks.eyebrows;
            }

            // Preserve a face mask separately for gating shader application
            if (generatedMasks.face) {
              filteredMasks.face = generatedMasks.face;
            } else if (generatedMasks.faceOval) {
              filteredMasks.face = generatedMasks.faceOval;
            }

            regionMasksRef.current = filteredMasks;

            // Update detected regions with only the active filters
            const detectedActiveRegions = [
              'skin',
              ...lipRegions,
              ...browRegions,
              'lips',
              'eyebrows',
            ].filter((region) => Boolean(filteredMasks[region]));
            setDetectedRegions(detectedActiveRegions);

            if (filteredMasks.face) {
              faceMaskRef.current = filteredMasks.face;
            } else {
              const canvas = document.createElement('canvas');
              canvas.width = canvasRef.current.width;
              canvas.height = canvasRef.current.height;
              const ctx = canvas.getContext('2d')!;
              ctx.fillStyle = 'black';
              ctx.fillRect(0, 0, canvas.width, canvas.height);
              faceMaskRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
            }
          }
        });

        if (mounted) {
          setIsInitialized(true);
          setIsPlaying(true);

          // Small delay to ensure everything is ready
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Start render loop
        let isRendering = true;
        const render = () => {
          if (!isRendering || !mounted) {
            return;
          }
          if (!canvasRef.current || !videoRef.current) {
            if (mounted) {
              animationFrameRef.current = requestAnimationFrame(render);
            }
            return;
          }

          // Check if video is ready
          if (videoRef.current.readyState < videoRef.current.HAVE_CURRENT_DATA) {
            if (mounted) {
              animationFrameRef.current = requestAnimationFrame(render);
            }
            return;
          }

          try {
            // Get current face mask - use 'face' or 'faceOval', fallback to black mask if no face detected
            let currentFaceMask: ImageData | undefined;
            if (faceMaskRef.current) {
              currentFaceMask = faceMaskRef.current;
            } else {
              // No face detected - create black mask so shader won't apply filter to background
              const canvas = document.createElement('canvas');
              canvas.width = canvasRef.current.width;
              canvas.height = canvasRef.current.height;
              const ctx = canvas.getContext('2d')!;
              ctx.fillStyle = 'black';
              ctx.fillRect(0, 0, canvas.width, canvas.height);
              currentFaceMask = ctx.getImageData(0, 0, canvas.width, canvas.height);
            }
            
            // Render frame with face mask
            renderer.render(
              videoRef.current,
              regionMasksRef.current,
              canvasRef.current.width,
              canvasRef.current.height,
              currentFaceMask,
              landmarksRef.current ?? undefined
            );
          } catch (err) {
            console.error('Render error:', err);
          }

          if (mounted && isPlaying && isRendering) {
            animationFrameRef.current = requestAnimationFrame(render);
          }
        };

        if (mounted && isPlaying) {
          render();
        }
        
        // Store cleanup function
        (window as any).filterCleanup = () => {
          isRendering = false;
        };
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Failed to initialize camera');
        setError(error.message);
        if (onError) {
          onError(error);
        }
      }
    };

    initialize();

    return () => {
      mounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      // Cleanup flag
      if (typeof window !== 'undefined' && (window as any).filterCleanup) {
        (window as any).filterCleanup();
        delete (window as any).filterCleanup;
      }
      
      // Cleanup async to avoid SSR issues
      if (typeof window !== 'undefined') {
        Promise.all([
          import('../services/mediapipe'),
          import('../services/webglRenderer')
        ]).then(([{ mediaPipeService }, { webglRenderer: renderer }]) => {
          mediaPipeService.stopCamera();
          renderer.cleanup();
        }).catch(err => {
          console.error('Cleanup error:', err);
        });
      }

      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [style, onError]);

  const handleStop = useCallback(() => {
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  }, []);

  const handleStart = useCallback(async () => {
    if (typeof window === 'undefined') return;
    
    setIsPlaying(true);
    const { webglRenderer: renderer } = await import('../services/webglRenderer');
    
    const render = () => {
      if (!canvasRef.current || !videoRef.current) {
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }

      try {
        renderer.render(
          videoRef.current,
          regionMasksRef.current,
          canvasRef.current.width,
          canvasRef.current.height,
          faceMaskRef.current ?? undefined,
          landmarksRef.current ?? undefined
        );
      } catch (err) {
        console.error('Render error:', err);
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };
    render();
  }, []);

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">Error: {error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="relative">
        <video
          ref={videoRef}
          className="hidden"
          playsInline
          muted
          autoPlay
        />
        <canvas
          ref={canvasRef}
          className="w-full max-w-2xl mx-auto rounded-lg shadow-lg"
          width={640}
          height={480}
        />
      </div>

      {isInitialized && (
        <div className="bg-white rounded-lg p-4 shadow">
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex gap-2">
                {!isPlaying ? (
                  <button
                    onClick={handleStart}
                    className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Start
                  </button>
                ) : (
                  <button
                    onClick={handleStop}
                    className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                  >
                    Stop
                  </button>
                )}
              </div>
              
              {/* Show detected regions */}
              {detectedRegions.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={() => setShowRegionFilters(!showRegionFilters)}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
                  >
                    {showRegionFilters ? 'Hide' : 'Show'} Region Filters
                  </button>
                  {showRegionFilters && (
                    <div className="mt-2 p-2 bg-gray-100 rounded text-sm">
                      <p className="font-semibold mb-1">Detected Regions:</p>
                      <div className="flex flex-wrap gap-1">
                        {detectedRegions.map(region => (
                          <span key={region} className="px-2 py-1 bg-blue-200 rounded text-xs">
                            {region}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

