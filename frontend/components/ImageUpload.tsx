'use client'

import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { CloudArrowUpIcon } from '@heroicons/react/24/outline'

interface ImageUploadProps {}

const ImageUpload: React.FC<ImageUploadProps> = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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

    try {
      // Create preview
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)

      // TODO: Upload to backend
      // const formData = new FormData()
      // formData.append('image', file)
      // const response = await fetch('/api/upload/image', {
      //   method: 'POST',
      //   body: formData,
      // })
      // const data = await response.json()
      
    } catch (err) {
      setError('Failed to upload image')
      console.error(err)
    } finally {
      setIsUploading(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  })

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

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {/* Image Preview */}
      {uploadedImage && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Uploaded Image</h3>
          <div className="relative">
            <img
              src={uploadedImage}
              alt="Uploaded"
              className="max-w-full h-auto rounded-lg shadow-md"
            />
          </div>
          <div className="text-center">
            <button
              onClick={() => setUploadedImage(null)}
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