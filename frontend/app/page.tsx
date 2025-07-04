import ImageUpload from '../components/ImageUpload'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Facetory
          </h1>
          <p className="text-xl text-gray-600">
            Create beautiful makeup filters from your photos using AI
          </p>
        </div>
        
        <div className="max-w-4xl mx-auto">
          <ImageUpload />
        </div>
      </div>
    </main>
  )
} 