import Link from 'next/link'
import ImageUpload from '../components/ImageUpload'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Facetory
          </h1>
          <p className="text-xl text-gray-600 mb-6">
            Create beautiful makeup filters from your photos using AI
          </p>
          
          <div className="flex gap-4 justify-center">
            <Link
              href="/filter"
              className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-semibold"
            >
              Real-Time Filter
            </Link>
            <Link
              href="/"
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold"
            >
              Style Analysis
            </Link>
          </div>
        </div>
        
        <div className="max-w-4xl mx-auto mt-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-2xl font-semibold mb-4">Upload Image for Style Analysis</h2>
            <ImageUpload />
          </div>
        </div>
      </div>
    </main>
  )
} 