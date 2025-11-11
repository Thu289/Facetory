"""
Migration script to convert old presigned URLs to proxy URLs
"""

import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

STYLES_DIR = Path("/tmp/facetory_styles")
STYLES_INDEX_FILE = STYLES_DIR / "styles_index.json"


def convert_url_to_proxy(url: str) -> str:
    """
    Convert MinIO presigned URL to proxy URL
    
    Example:
    minio:9000/facetory-storage/styles/xxx/luts/xxx.bin?params...
    -> /api/makeup/storage/file/styles/xxx/luts/xxx.bin
    """
    if not url or url.startswith('/api/'):
        return url  # Already proxy URL or empty
    
    try:
        # Parse URL
        parsed = urlparse(url)
        path = parsed.path
        
        # Remove bucket prefix if present
        if '/facetory-storage/' in path:
            path = path.split('/facetory-storage/', 1)[1]
        elif path.startswith('/'):
            path = path[1:]
        
        # Encode object name
        encoded = quote(path, safe='')
        
        # Return proxy URL
        return f"/api/makeup/storage/file/{encoded}"
    except Exception as e:
        print(f"Error converting URL {url}: {e}")
        return url


def migrate_style_urls(style_data: dict) -> dict:
    """Migrate URLs in style data"""
    if 'download_urls' not in style_data:
        return style_data
    
    download_urls = style_data['download_urls']
    
    # Convert region mask URLs
    if 'region_masks' in download_urls:
        for region, url in download_urls['region_masks'].items():
            download_urls['region_masks'][region] = convert_url_to_proxy(url)
    
    # Convert other URLs
    for key in ['style_parameters', 'thumbnail']:
        if key in download_urls and download_urls[key]:
            download_urls[key] = convert_url_to_proxy(download_urls[key])
    
    return style_data


def migrate_all_styles():
    """Migrate all styles in storage"""
    if not STYLES_INDEX_FILE.exists():
        print("No styles to migrate")
        return
    
    with open(STYLES_INDEX_FILE, 'r') as f:
        index = json.load(f)
    
    migrated = 0
    for style_entry in index.get('styles', []):
        style_id = style_entry['style_id']
        style_file = STYLES_DIR / f"{style_id}.json"
        
        if not style_file.exists():
            continue
        
        try:
            with open(style_file, 'r') as f:
                style_data = json.load(f)
            
            # Migrate URLs
            migrated_data = migrate_style_urls(style_data.copy())
            
            # Save if changed
            if migrated_data != style_data:
                with open(style_file, 'w') as f:
                    json.dump(migrated_data, f, indent=2)
                migrated += 1
                print(f"✅ Migrated {style_id}")
        except Exception as e:
            print(f"❌ Error migrating {style_id}: {e}")
    
    print(f"Migration complete: {migrated} styles migrated")


if __name__ == "__main__":
    migrate_all_styles()

