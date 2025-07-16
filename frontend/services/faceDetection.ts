export interface FaceDetectionResult {
  num_faces: number;
  faces: {
    face_id: string;
    bounding_box: [number, number, number, number];
    landmarks: Record<string, [number, number]>;
  }[];
  image_size: { width: number; height: number };
}

export async function detectFaces(file: File): Promise<FaceDetectionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/face/detect', {
    method: 'POST',
    body: formData,
  });
  const text = await res.text();
  console.log('API response:', text);
  if (!res.ok) {
    throw new Error('Face detection failed: ' + text);
  }
  return JSON.parse(text);
}

export interface MakeupExtractionResult {
  lips_color: [number, number, number]; // [R, G, B]
  left_eye_color: [number, number, number];
  right_eye_color: [number, number, number];
}

export async function extractMakeup(file: File | Blob): Promise<MakeupExtractionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/makeup/extract', {
    method: 'POST',
    body: formData,
  });
  const text = await res.text();
  console.log('Makeup API response:', text);
  if (!res.ok) {
    throw new Error('Makeup extraction failed: ' + text);
  }
  return JSON.parse(text);
} 