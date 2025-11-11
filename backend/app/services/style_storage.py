"""
Simple file-based style storage for development
TODO: Replace with database storage in production
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path

STYLES_DIR = Path("/tmp/facetory_styles")
STYLES_INDEX_FILE = STYLES_DIR / "styles_index.json"


def ensure_styles_dir():
    """Ensure styles directory exists"""
    STYLES_DIR.mkdir(parents=True, exist_ok=True)
    if not STYLES_INDEX_FILE.exists():
        with open(STYLES_INDEX_FILE, 'w') as f:
            json.dump({"styles": []}, f)


def save_style(style_id: str, style_data: Dict) -> bool:
    """Save style data to file"""
    try:
        ensure_styles_dir()
        
        # Save style data
        style_file = STYLES_DIR / f"{style_id}.json"
        with open(style_file, 'w') as f:
            json.dump(style_data, f, indent=2)
        
        # Update index
        with open(STYLES_INDEX_FILE, 'r') as f:
            index = json.load(f)
        
        # Check if style_id already exists
        existing_index = next((i for i, s in enumerate(index["styles"]) if s["style_id"] == style_id), None)
        
        style_entry = {
            "style_id": style_id,
            "name": style_data.get("name", f"Style {style_id}"),
            "description": style_data.get("description"),
            "created_at": style_data.get("created_at", "unknown")
        }
        
        if existing_index is not None:
            index["styles"][existing_index] = style_entry
        else:
            index["styles"].append(style_entry)
        
        with open(STYLES_INDEX_FILE, 'w') as f:
            json.dump(index, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving style: {e}")
        return False


def get_style(style_id: str) -> Optional[Dict]:
    """Get style data by ID"""
    try:
        ensure_styles_dir()
        style_file = STYLES_DIR / f"{style_id}.json"
        
        if not style_file.exists():
            return None
        
        with open(style_file, 'r') as f:
            style_data = json.load(f)
        
        # Migrate old URLs to proxy URLs if needed
        from urllib.parse import urlparse, quote
        if 'download_urls' in style_data:
            download_urls = style_data['download_urls']
            changed = False
            
            # Convert region mask URLs
            if 'region_masks' in download_urls:
                for region, url in download_urls['region_masks'].items():
                    if url and not url.startswith('/api/') and ('minio:' in url or urlparse(url).hostname == 'minio'):
                        # Extract object name from URL
                        path = urlparse(url).path
                        if '/facetory-storage/' in path:
                            obj_name = path.split('/facetory-storage/', 1)[1]
                        else:
                            obj_name = path.lstrip('/')
                        download_urls['region_masks'][region] = f"/api/makeup/storage/file/{quote(obj_name, safe='')}"
                        changed = True
            
            # Save migrated data
            if changed:
                with open(style_file, 'w') as f:
                    json.dump(style_data, f, indent=2)
        
        return style_data
    except Exception as e:
        print(f"Error loading style: {e}")
        return None


def list_styles(limit: int = 20, offset: int = 0) -> Dict:
    """List all styles"""
    try:
        ensure_styles_dir()
        
        if not STYLES_INDEX_FILE.exists():
            return {"styles": [], "total": 0}
        
        with open(STYLES_INDEX_FILE, 'r') as f:
            index = json.load(f)
        
        styles = index.get("styles", [])
        total = len(styles)
        
        # Apply pagination
        paginated = styles[offset:offset + limit]
        
        return {
            "styles": paginated,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        print(f"Error listing styles: {e}")
        return {"styles": [], "total": 0}

