#!/usr/bin/env python3
"""
Test script to apply filter to an image and save the result
Usage: python test_apply_filter.py <image_path> <style_id> [output_path]
"""

import sys
import os
import requests
from pathlib import Path

def test_apply_filter(image_path: str, style_id: str, output_path: str = None, intensity: float = 1.0):
    """
    Test the apply_filter API endpoint
    
    Args:
        image_path: Path to input image
        style_id: Style ID to apply
        output_path: Optional output path for filtered image
        intensity: Filter intensity (0.0 to 1.0)
    """
    api_url = "http://localhost:8000/api/makeup/style/apply_filter"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return False
    
    # Prepare form data
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {
            'style_id': style_id,
            'intensity': str(intensity)
        }
        
        print(f"🔄 Sending request to {api_url}...")
        print(f"   Image: {image_path}")
        print(f"   Style ID: {style_id}")
        print(f"   Intensity: {intensity}")
        
        try:
            response = requests.post(api_url, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    print("✅ Filter applied successfully!")
                    
                    # Decode base64 image
                    import base64
                    filtered_image_data = result['filtered_image']
                    
                    # Extract base64 data
                    if filtered_image_data.startswith('data:image'):
                        # Format: data:image/jpeg;base64,<base64_data>
                        base64_data = filtered_image_data.split(',', 1)[1]
                    else:
                        base64_data = filtered_image_data
                    
                    # Decode
                    image_bytes = base64.b64decode(base64_data)
                    
                    # Save to file
                    if output_path is None:
                        input_name = Path(image_path).stem
                        output_path = f"{input_name}_filtered_{style_id}.jpg"
                    
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    print(f"💾 Filtered image saved to: {output_path}")
                    print(f"   Original size: {result.get('original_size', 'N/A')}")
                    
                    return True
                else:
                    print(f"❌ API returned success=False: {result}")
                    return False
            else:
                print(f"❌ API Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Detail: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text[:200]}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_apply_filter.py <image_path> <style_id> [output_path] [intensity]")
        print("\nExample:")
        print("  python test_apply_filter.py test.jpg style_a66c305c")
        print("  python test_apply_filter.py test.jpg style_a66c305c output.jpg 0.8")
        sys.exit(1)
    
    image_path = sys.argv[1]
    style_id = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    intensity = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    
    success = test_apply_filter(image_path, style_id, output_path, intensity)
    sys.exit(0 if success else 1)

