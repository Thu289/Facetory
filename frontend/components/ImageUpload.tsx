'use client'

import React, { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { CloudArrowUpIcon } from '@heroicons/react/24/outline'
import { detectFaces, FaceDetectionResult, extractMakeup, MakeupExtractionResult } from '../services/faceDetection'

interface ImageUploadProps {}

const ImageUpload: React.FC<ImageUploadProps> = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [boundingBoxes, setBoundingBoxes] = useState<{
    box: [number, number, number, number];
    face_id: string;
  }[]>([])
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null)
  const [detecting, setDetecting] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)
  const [displaySize, setDisplaySize] = useState<{ width: number; height: number } | null>(null)
  const [selectedFace, setSelectedFace] = useState<number | null>(null)
  const [croppedImage, setCroppedImage] = useState<string | null>(null)
  const [makeupResult, setMakeupResult] = useState<MakeupExtractionResult | null>(null)
  const [extractingMakeup, setExtractingMakeup] = useState(false)
  const [makeupError, setMakeupError] = useState<string | null>(null)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file')
      return
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB')
      return
    }

    setIsUploading(true)
    setError(null)
    setBoundingBoxes([])
    setImageSize(null)
    setUploadedFile(file)

    try {
      // Create preview
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)

      // Gọi detect face API
      setDetecting(true)
      console.log('Detecting faces...')
      const detection: FaceDetectionResult = await detectFaces(file)
      console.log('Detection:', detection)
      setBoundingBoxes(
        detection.faces.map((f) => ({
          box: f.bounding_box,
          face_id: f.face_id,
        }))
      )
      setImageSize(detection.image_size)
    } catch (err) {
      setError('Failed to detect face')
      setBoundingBoxes([])
      setImageSize(null)
      console.error(err)
    } finally {
      setIsUploading(false)
      setDetecting(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  })

  // Lấy kích thước hiển thị thực tế của ảnh preview
  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.target as HTMLImageElement
    setDisplaySize({ width: img.width, height: img.height })
  }

  // Tính scale giữa ảnh gốc và ảnh preview
  const getScale = () => {
    if (!imageSize || !displaySize) return { x: 1, y: 1 }
    return {
      x: displaySize.width / imageSize.width,
      y: displaySize.height / imageSize.height,
    }
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive 
            ? 'border-primary-500 bg-primary-50' 
            : 'border-gray-300 hover:border-gray-400'
          }
        `}
      >
        <input {...getInputProps()} />
        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        {isDragActive ? (
          <p className="text-primary-600">Drop the image here...</p>
        ) : (
          <div>
            <p className="text-gray-600 mb-2">
              Drag & drop an image here, or click to select
            </p>
            <p className="text-sm text-gray-500">
              Supports JPG, PNG up to 10MB
            </p>
          </div>
        )}
      </div>

      {/* Loading State */}
      {isUploading && (
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
          <p className="mt-2 text-gray-600">Uploading image...</p>
        </div>
      )}
      {detecting && (
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
          <p className="mt-2 text-gray-600">Detecting faces...</p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {/* Image Preview + Bounding Box Overlay */}
      {uploadedImage && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Uploaded Image</h3>
          <div className="relative inline-block">
            <img
              src={uploadedImage}
              alt="Uploaded"
              className="max-w-full h-auto rounded-lg shadow-md"
              ref={imgRef}
              onLoad={handleImageLoad}
              style={{ display: 'block' }}
            />
            {/* Overlay bounding boxes */}
            {boundingBoxes.map(({ box, face_id }, idx) => {
              if (!imageSize || !displaySize) return null
              const [x1, y1, x2, y2] = box
              const scale = getScale()
              const left = x1 * scale.x
              const top = y1 * scale.y
              const width = (x2 - x1) * scale.x
              const height = (y2 - y1) * scale.y
              return (
                <div
                  key={face_id}
                  className={`absolute border-2 ${selectedFace === idx ? 'border-blue-500' : 'border-red-500'} cursor-pointer`}
                  style={{ left, top, width, height, pointerEvents: 'auto' }}
                  onClick={() => setSelectedFace(idx)}
                  title="Click to select this face"
                />
              )
            })}
          </div>
          {boundingBoxes.length > 0 && (
            <div className="mt-2 text-green-600 font-semibold">
              {boundingBoxes.length} face(s) detected!
            </div>
          )}
          {selectedFace !== null && (
            <button
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
              onClick={async () => {
                if (!uploadedFile || !boundingBoxes[selectedFace]) return
                const [x1, y1, x2, y2] = boundingBoxes[selectedFace].box
                const formData = new FormData()
                formData.append('file', uploadedFile)
                formData.append('x1', x1.toString())
                formData.append('y1', y1.toString())
                formData.append('x2', x2.toString())
                formData.append('y2', y2.toString())
                const res = await fetch('/api/face/crop', {
                  method: 'POST',
                  body: formData,
                })
                const data = await res.json()
                setCroppedImage('data:image/jpeg;base64,' + data.cropped_image_base64)
              }}
            >
              Crop Selected Face
            </button>
          )}
          {croppedImage && (
            <div className="mt-4">
              <h4 className="font-semibold">Cropped Face</h4>
              <img src={croppedImage} alt="Cropped face" className="rounded shadow" />
              <div className="mt-2">
                <button
                  className="px-4 py-2 bg-purple-600 text-white rounded"
                  onClick={async () => {
                    setExtractingMakeup(true)
                    setMakeupError(null)
                    setMakeupResult(null)
                    try {
                      // Convert base64 to Blob
                      const res = await fetch(croppedImage)
                      const blob = await res.blob()
                      const result = await extractMakeup(blob)
                      setMakeupResult(result)
                    } catch (err: any) {
                      setMakeupError(err.message || 'Failed to extract makeup')
                    } finally {
                      setExtractingMakeup(false)
                    }
                  }}
                  disabled={extractingMakeup}
                >
                  {extractingMakeup ? 'Extracting Makeup...' : 'Extract Makeup Attributes'}
                </button>
                {makeupError && (
                  <div className="mt-2 text-red-600">{makeupError}</div>
                )}
                {makeupResult && (
                  <div className="mt-4 space-y-2">
                    <div>
                      <span className="font-semibold">Lips Color:</span>
                      <span
                        className="inline-block w-6 h-6 ml-2 rounded align-middle border"
                        style={{ background: `rgb(${makeupResult.lips_color.join(',')})` }}
                      ></span>
                      <span className="ml-2 text-sm text-gray-700">[{makeupResult.lips_color.join(', ')}]</span>
                    </div>
                    <div>
                      <span className="font-semibold">Left Eye Color:</span>
                      <span
                        className="inline-block w-6 h-6 ml-2 rounded align-middle border"
                        style={{ background: `rgb(${makeupResult.left_eye_color.join(',')})` }}
                      ></span>
                      <span className="ml-2 text-sm text-gray-700">[{makeupResult.left_eye_color.join(', ')}]</span>
                    </div>
                    <div>
                      <span className="font-semibold">Right Eye Color:</span>
                      <span
                        className="inline-block w-6 h-6 ml-2 rounded align-middle border"
                        style={{ background: `rgb(${makeupResult.right_eye_color.join(',')})` }}
                      ></span>
                      <span className="ml-2 text-sm text-gray-700">[{makeupResult.right_eye_color.join(', ')}]</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
          <div className="text-center">
            <button
              onClick={() => {
                setUploadedImage(null)
                setBoundingBoxes([])
                setImageSize(null)
                setDisplaySize(null)
                setUploadedFile(null)
              }}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
            >
              Upload Different Image
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default ImageUpload 