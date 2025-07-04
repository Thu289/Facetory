-- Create database if not exists
-- This will be handled by Docker environment variables

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    username VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create original_images table
CREATE TABLE IF NOT EXISTS original_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    filename VARCHAR(255) NOT NULL,
    minio_path VARCHAR(500) NOT NULL,
    face_count INTEGER DEFAULT 1,
    image_size INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create makeup_extractions table
CREATE TABLE IF NOT EXISTS makeup_extractions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_image_id UUID REFERENCES original_images(id),
    lip_color VARCHAR(7),
    eye_shadow_color VARCHAR(7),
    blush_color VARCHAR(7),
    brow_shape VARCHAR(50),
    makeup_attributes JSONB,
    landmarks JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create filters table
CREATE TABLE IF NOT EXISTS filters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    makeup_extraction_id UUID REFERENCES makeup_extractions(id),
    filter_name VARCHAR(100),
    filter_data JSONB,
    minio_path VARCHAR(500),
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create captured_photos table
CREATE TABLE IF NOT EXISTS captured_photos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    filter_id UUID REFERENCES filters(id),
    filename VARCHAR(255) NOT NULL,
    minio_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_original_images_user_id ON original_images(user_id);
CREATE INDEX IF NOT EXISTS idx_filters_user_id ON filters(user_id);
CREATE INDEX IF NOT EXISTS idx_captured_photos_user_id ON captured_photos(user_id); 