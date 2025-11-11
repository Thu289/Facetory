"""
API endpoints for model information
"""

from fastapi import APIRouter
from pathlib import Path
import os
import torch
from typing import Dict, Any

router = APIRouter()


@router.get("/model/bisenet/info")
async def get_bisenet_model_info():
    """
    Get information about BiSeNet model including:
    - Model weights path and existence
    - Model architecture
    - Repository status
    - Device information
    """
    from ai_models.BiseNet.inference_bisenet import (
        BISENET_AVAILABLE,
        BISENET_IMPORT_ERROR,
        NUM_CLASSES,
        CELEBA_ATTRIBUTES
    )
    
    info: Dict[str, Any] = {
        "model_name": "BiSeNet Face Parsing",
        "num_classes": NUM_CLASSES,
        "attributes": CELEBA_ATTRIBUTES,
        "bisenet_available": BISENET_AVAILABLE,
        "import_error": BISENET_IMPORT_ERROR,
        "weights": {},
        "repository": {},
        "device": {}
    }
    
    # Check weights paths
    base_path = Path(__file__).parent.parent.parent / "ai_models" / "BiseNet"
    repo_weights = base_path / "face-parsing.PyTorch" / "res" / "cp" / "79999_iter.pth"
    fallback_weights = base_path / "pretrained_models" / "bisenet" / "79999_iter.pth"
    
    info["weights"] = {
        "repo_path": str(repo_weights),
        "repo_exists": repo_weights.exists(),
        "fallback_path": str(fallback_weights),
        "fallback_exists": fallback_weights.exists(),
        "selected_path": None,
        "selected_exists": False
    }
    
    if repo_weights.exists():
        info["weights"]["selected_path"] = str(repo_weights)
        info["weights"]["selected_exists"] = True
    elif fallback_weights.exists():
        info["weights"]["selected_path"] = str(fallback_weights)
        info["weights"]["selected_exists"] = True
    
    # Get weights file size if exists
    if info["weights"]["selected_path"]:
        weights_path = Path(info["weights"]["selected_path"])
        if weights_path.exists():
            size_mb = weights_path.stat().st_size / (1024 * 1024)
            info["weights"]["file_size_mb"] = round(size_mb, 2)
    
    # Check repository
    repo_path = base_path / "face-parsing.PyTorch"
    info["repository"] = {
        "path": str(repo_path),
        "exists": repo_path.exists(),
        "model_py_exists": (repo_path / "model.py").exists() if repo_path.exists() else False,
        "resnet_py_exists": (repo_path / "resnet.py").exists() if repo_path.exists() else False,
    }
    
    # Device info
    info["device"] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Try to load model and get architecture info
    try:
        if info["weights"]["selected_exists"]:
            from ai_models.BiseNet.inference_bisenet import load_bisenet_model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_bisenet_model(weights_path=info["weights"]["selected_path"], device=device)
            
            if model is not None:
                info["model_loaded"] = True
                info["model_architecture"] = {
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "device": str(next(model.parameters()).device),
                    "model_type": type(model).__name__
                }
            else:
                info["model_loaded"] = False
                info["model_load_error"] = "Model failed to load"
        else:
            info["model_loaded"] = False
            info["model_load_error"] = "Weights file not found"
    except Exception as e:
        info["model_loaded"] = False
        info["model_load_error"] = str(e)
    
    return info


@router.get("/model/bisenet/weights/check")
async def check_bisenet_weights():
    """
    Check BiSeNet weights file and display structure
    """
    from pathlib import Path
    import torch
    
    base_path = Path(__file__).parent.parent.parent / "ai_models" / "BiseNet"
    repo_weights = base_path / "face-parsing.PyTorch" / "res" / "cp" / "79999_iter.pth"
    fallback_weights = base_path / "pretrained_models" / "bisenet" / "79999_iter.pth"
    
    weights_path = None
    if repo_weights.exists():
        weights_path = repo_weights
    elif fallback_weights.exists():
        weights_path = fallback_weights
    
    if not weights_path or not weights_path.exists():
        return {
            "error": "Weights file not found",
            "checked_paths": [
                str(repo_weights),
                str(fallback_weights)
            ]
        }
    
    try:
        # Load weights
        weights = torch.load(weights_path, map_location='cpu')
        
        # Get structure info
        if isinstance(weights, dict):
            keys = list(weights.keys())
            info = {
                "file_path": str(weights_path),
                "file_size_mb": round(weights_path.stat().st_size / (1024 * 1024), 2),
                "state_dict_type": type(weights).__name__,
                "num_keys": len(keys),
                "sample_keys": keys[:10],  # First 10 keys
                "total_keys": keys,
                "key_prefixes": list(set([k.split('.')[0] for k in keys if '.' in k]))
            }
            
            # Check for common BiSeNet key patterns
            info["has_cp_prefix"] = any(k.startswith('cp.') for k in keys)
            info["has_resnet"] = any('resnet' in k.lower() for k in keys)
            
        else:
            info = {
                "file_path": str(weights_path),
                "file_size_mb": round(weights_path.stat().st_size / (1024 * 1024), 2),
                "state_dict_type": type(weights).__name__,
                "error": "Weights is not a dictionary"
            }
        
        return info
        
    except Exception as e:
        return {
            "error": f"Failed to load weights: {str(e)}",
            "file_path": str(weights_path)
        }

