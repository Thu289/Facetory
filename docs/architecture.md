sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant RetinaFace
    participant BiSeNet
    participant StyleExtractor
    participant LUTGen
    participant ShaderGen
    participant Storage
    
    User->>Frontend: Upload Image
    Frontend->>API: POST /api/makeup/style/create_complete
    API->>RetinaFace: Detect Face
    RetinaFace-->>API: Face Bounding Box
    API->>BiSeNet: Segment Face Regions
    BiSeNet-->>API: Segmentation Mask (19 classes)
    API->>StyleExtractor: Extract Style Parameters
    StyleExtractor->>StyleExtractor: RGB to LAB conversion
    StyleExtractor->>StyleExtractor: K-means clustering
    StyleExtractor->>StyleExtractor: Histogram analysis
    StyleExtractor-->>API: Style Parameters (JSON)
    API->>LUTGen: Generate 3D LUTs
    LUTGen-->>API: LUT files (binary)
    API->>ShaderGen: Generate WebGL Shaders
    ShaderGen-->>API: Shader files (GLSL)
    API->>Storage: Upload Assets
    Storage-->>API: Download URLs
    API-->>Frontend: Style ID + URLs
    Frontend-->>User: Style Created
