# Facetory - Architecture Diagram

## System Architecture Overview

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer (Browser)"
        A[React/Next.js Frontend]
        B[WebRTC Camera]
        C[MediaPipe FaceMesh]
        D[WebGL Renderer]
        E[LUT Loader]
    end
    
    subgraph "API Gateway"
        F[FastAPI Server]
        G[REST API Endpoints]
    end
    
    subgraph "AI Processing Layer"
        H[RetinaFace<br/>Face Detection]
        I[BiSeNet<br/>Face Segmentation]
        J[Style Extraction<br/>LAB + K-means]
        K[LUT Generator]
        L[Shader Generator]
    end
    
    subgraph "Storage Layer"
        M[MinIO<br/>Object Storage]
        N[PostgreSQL<br/>Database]
        O[Redis<br/>Cache]
    end
    
    A --> B
    A --> C
    A --> D
    A --> E
    B --> F
    C --> D
    D --> E
    A --> F
    F --> G
    G --> H
    G --> I
    G --> J
    J --> K
    J --> L
    K --> M
    L --> M
    F --> M
    F --> N
    F --> O
    E --> M
```

## Detailed Component Architecture

### Phase 1: Style Creation Pipeline

```mermaid
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
```

### Phase 2: Real-Time Filter Application

```mermaid
sequenceDiagram
    participant User
    participant Camera
    participant MediaPipe
    participant WebGL
    participant Shader
    participant LUT
    
    User->>Camera: Start Camera
    Camera->>MediaPipe: Video Stream
    MediaPipe->>MediaPipe: Detect Face (468 landmarks)
    MediaPipe->>MediaPipe: Generate Region Masks
    MediaPipe->>WebGL: Region Masks (RGBA)
    Camera->>WebGL: Video Frame
    WebGL->>LUT: Load LUTs from URLs
    LUT-->>WebGL: LUT Textures
    WebGL->>Shader: Apply Filter
    Shader->>Shader: Face Mask Check (threshold 0.15)
    Shader->>Shader: Region Mask Blending
    Shader->>Shader: LUT Color Lookup
    Shader->>Shader: Intensity Blending
    Shader-->>WebGL: Filtered Frame
    WebGL-->>User: Display Result (30-60 FPS)
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Frontend Components"
        A[ImageUpload]
        B[StyleSelector]
        C[CameraFilter]
        D[StyleUpload]
    end
    
    subgraph "Frontend Services"
        E[API Service]
        F[MediaPipe Service]
        G[WebGL Renderer]
        H[LUT Loader]
    end
    
    subgraph "Backend API"
        I[Style Management API]
        J[Face Detection API]
        K[Upload API]
    end
    
    subgraph "Backend Services"
        L[Style Extraction]
        M[LUT Generation]
        N[Shader Generation]
        O[Storage Service]
    end
    
    subgraph "AI Models"
        P[RetinaFace]
        Q[BiSeNet]
    end
    
    A --> E
    B --> E
    C --> F
    C --> G
    D --> E
    E --> I
    E --> J
    E --> K
    I --> L
    I --> M
    I --> N
    I --> O
    J --> P
    L --> Q
    G --> H
    H --> O
```

## Data Flow Architecture

```mermaid
flowchart TD
    Start([User Action]) --> Upload{Upload Image?}
    Upload -->|Yes| Phase1[Phase 1: Style Creation]
    Upload -->|No| Phase2[Phase 2: Real-Time Filter]
    
    Phase1 --> P1A[1. Image Upload]
    P1A --> P1B[2. RetinaFace Detection]
    P1B --> P1C[3. BiSeNet Segmentation]
    P1C --> P1D[4. Style Extraction]
    P1D --> P1E[5. LUT Generation]
    P1E --> P1F[6. Shader Generation]
    P1F --> P1G[7. Storage Upload]
    P1G --> P1H[8. Return Style ID]
    
    Phase2 --> P2A[1. Select Style]
    P2A --> P2B[2. Load LUTs & Shaders]
    P2B --> P2C[3. Initialize Camera]
    P2C --> P2D[4. Initialize MediaPipe]
    P2D --> P2E[5. Initialize WebGL]
    P2E --> P2F[6. Render Loop]
    P2F --> P2G[7. Apply Filter]
    P2G --> P2H[8. Display Result]
    P2H --> P2F
    
    style Phase1 fill:#e1f5ff
    style Phase2 fill:#fff4e1
    style P1G fill:#c8e6c9
    style P2H fill:#c8e6c9
```

## Storage Architecture

```mermaid
graph TB
    subgraph "Storage Services"
        A[MinIO Object Storage]
        B[PostgreSQL Database]
        C[Redis Cache]
    end
    
    subgraph "Stored Assets"
        D[LUT Files<br/>32x32x32 3D LUTs<br/>Binary format]
        E[Shader Files<br/>GLSL code<br/>Fragment & Vertex]
        F[Style Metadata<br/>JSON parameters]
        G[Thumbnails<br/>Preview images]
    end
    
    subgraph "Database Tables"
        H[users]
        I[styles]
        J[style_assets]
        K[original_images]
    end
    
    A --> D
    A --> E
    A --> G
    B --> H
    B --> I
    B --> J
    B --> K
    C --> F
    
    I --> A
    J --> A
```

## Technology Stack Diagram

```mermaid
graph TB
    subgraph "Frontend Stack"
        A1[React/Next.js]
        A2[TypeScript]
        A3[WebRTC]
        A4[WebGL]
        A5[MediaPipe]
    end
    
    subgraph "Backend Stack"
        B1[FastAPI]
        B2[Python 3.9+]
        B3[Docker]
        B4[PostgreSQL]
        B5[MinIO]
        B6[Redis]
    end
    
    subgraph "AI/ML Stack"
        C1[PyTorch]
        C2[RetinaFace]
        C3[BiSeNet]
        C4[OpenCV]
        C5[scikit-learn]
        C6[NumPy]
    end
    
    subgraph "Infrastructure"
        D1[Docker Compose]
        D2[Nginx]
        D3[GPU Support]
    end
    
    A1 --> A2
    A1 --> A3
    A1 --> A4
    A1 --> A5
    B1 --> B2
    B1 --> B3
    B1 --> B4
    B1 --> B5
    B1 --> B6
    C1 --> C2
    C1 --> C3
    C2 --> C4
    C3 --> C4
    C4 --> C5
    C4 --> C6
```

## Performance Optimization Architecture

```mermaid
graph TB
    subgraph "Client-Side Optimization"
        A1[GPU Acceleration<br/>WebGL]
        A2[LUT Caching<br/>Browser Memory]
        A3[Shader Compilation<br/>Caching]
        A4[Frame Throttling<br/>30-60 FPS]
        A5[Efficient Face Tracking<br/>MediaPipe]
    end
    
    subgraph "Server-Side Optimization"
        B1[Batch Processing]
        B2[Async Processing]
        B3[CDN Distribution]
        B4[Asset Compression]
    end
    
    subgraph "Storage Optimization"
        C1[Binary LUT Format]
        C2[Minified Shaders]
        C3[Progressive Loading]
        C4[Redis Caching]
    end
    
    A1 --> A2
    A1 --> A3
    A1 --> A4
    A1 --> A5
    B1 --> B2
    B2 --> B3
    B3 --> B4
    C1 --> C2
    C2 --> C3
    C3 --> C4
```

## Security Architecture

```mermaid
graph TB
    subgraph "Authentication Layer"
        A1[JWT Tokens]
        A2[Password Hashing<br/>bcrypt]
        A3[Session Management]
    end
    
    subgraph "Authorization Layer"
        B1[User Roles]
        B2[Resource Ownership]
        B3[API Permissions]
    end
    
    subgraph "Data Protection"
        C1[File Upload Validation]
        C2[Image Size Limits]
        C3[Sanitization]
    end
    
    subgraph "Network Security"
        D1[HTTPS/TLS]
        D2[CORS Policies]
        D3[Rate Limiting]
    end
    
    A1 --> B1
    A2 --> B2
    A3 --> B3
    C1 --> C2
    C2 --> C3
    D1 --> D2
    D2 --> D3
```

