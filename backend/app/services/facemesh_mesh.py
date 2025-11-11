"""
Utilities for extracting MediaPipe FaceMesh-based region meshes for makeup application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay


# FaceMesh landmark index selections for regions
LIPS_INDICES = sorted(
    {
        0,
        13,
        14,
        17,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        78,
        80,
        81,
        82,
        84,
        87,
        88,
        91,
        95,
        178,
        181,
        185,
        191,
        268,
        271,
        272,
        273,
        274,
        275,
        276,
        291,
        292,
        293,
        294,
        295,
        296,
        302,
        303,
        304,
        308,
        312,
        313,
        314,
        317,
        318,
        321,
        324,
        402,
        405,
        407,
        415,
        416,
        425,
        428,
        429,
        436,
        437,
        438,
    }
)

LIPS_UPPER_INDICES = sorted(
    {
        idx
        for idx in LIPS_INDICES
        if idx
        in {
            0,
            13,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            76,
            78,
            80,
            81,
            82,
            84,
            87,
            181,
            185,
            191,
            271,
            272,
            273,
            274,
            275,
            276,
            291,
            292,
            293,
            294,
            324,
            402,
            415,
        }
    }
)

LIPS_LOWER_INDICES = sorted(
    set(LIPS_INDICES) - set(LIPS_UPPER_INDICES)
)

LEFT_EYEBROW_INDICES = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55, 46, 124, 35, 227]
RIGHT_EYEBROW_INDICES = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 295, 413, 265, 353]

NOSE_INDICES = [
    4,
    51,
    48,
    115,
    131,
    134,
    102,
    49,
    98,
    97,
    326,
    327,
    358,
    359,
    420,
    279,
    309,
    429,
    279,
    330,
    347,
    348,
    94,
    331,
    278,
    118,
]

FACE_OVAL_INDICES = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    127,
    234,
    93,
    132,
    58,
]

CHEEK_INDICES = [
    50,
    205,
    50,
    101,
    100,
    47,
    187,
    147,
    213,
    192,
    203,
    129,
    208,
    171,
    32,
    211,
    210,
    204,
    212,
    57,
    43,
    106,
    63,
    105,
    104,
    55,
    8,
    285,
    417,
    441,
    442,
    443,
    444,
    413,
    456,
    399,
]


REGION_LANDMARK_MAP: Dict[str, Sequence[int]] = {
    "lips": LIPS_INDICES,
    "lips_upper": LIPS_UPPER_INDICES,
    "lips_lower": LIPS_LOWER_INDICES,
    "eyebrows": list(dict.fromkeys(LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES)),
    "eyebrow_left": LEFT_EYEBROW_INDICES,
    "eyebrow_right": RIGHT_EYEBROW_INDICES,
    "nose": list(dict.fromkeys(NOSE_INDICES)),
    "skin": list(dict.fromkeys(FACE_OVAL_INDICES + CHEEK_INDICES + NOSE_INDICES)),
}


@dataclass
class RegionMesh:
    image_width: int
    image_height: int
    points: List[Dict[str, float]]
    triangles: List[Tuple[int, int, int]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_size": [self.image_width, self.image_height],
            "points": self.points,
            "triangles": [list(tri) for tri in self.triangles],
        }


def _unique_indices(indices: Iterable[int]) -> List[int]:
    seen = set()
    result: List[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def _compute_triangles(points_xy: np.ndarray) -> Optional[List[Tuple[int, int, int]]]:
    if points_xy.shape[0] < 3:
        return None

    try:
        delaunay = Delaunay(points_xy)
    except Exception:
        return None

    simplices = []
    for simplex in delaunay.simplices:
        if len(simplex) == 3:
            simplices.append(tuple(int(i) for i in simplex))
    return simplices if simplices else None


def _extract_face_landmarks(image_rgb: np.ndarray) -> Optional[Tuple[List[mp.framework.formats.landmark_pb2.NormalizedLandmark], int, int]]:
    height, width = image_rgb.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark, width, height


def compute_region_meshes(image_rgb: np.ndarray) -> Dict[str, Dict[str, object]]:
    """
    Compute region meshes for the given RGB face image.

    Returns:
        Mapping of region name -> mesh dict with fields:
            - image_size: [width, height]
            - points: [{ "index": int, "x": float, "y": float }]
            - triangles: [[i1, i2, i3], ...]
    """
    extraction = _extract_face_landmarks(image_rgb)
    if extraction is None:
        return {}

    landmarks, width, height = extraction

    region_meshes: Dict[str, Dict[str, object]] = {}

    for region_name, indices in REGION_LANDMARK_MAP.items():
        unique_indices = _unique_indices(indices)
        pts: List[Dict[str, float]] = []
        coords_xy: List[Tuple[float, float]] = []

        for idx in unique_indices:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            pts.append(
                {
                    "index": int(idx),
                    "x": float(lm.x),
                    "y": float(lm.y),
                }
            )
            coords_xy.append((float(lm.x) * width, float(lm.y) * height))

        if len(pts) < 3:
            continue

        triangles = _compute_triangles(np.array(coords_xy, dtype=np.float32))
        if not triangles:
            continue

        mesh = RegionMesh(
            image_width=width,
            image_height=height,
            points=pts,
            triangles=triangles,
        )
        region_meshes[region_name] = mesh.to_dict()

    return region_meshes


def extract_normalized_landmarks(image_rgb: np.ndarray) -> Optional[Dict[int, Tuple[float, float]]]:
    """
    Extract normalized (x, y) landmarks for the provided RGB image.
    """
    extraction = _extract_face_landmarks(image_rgb)
    if extraction is None:
        return None

    landmarks, _, _ = extraction
    return {idx: (float(lm.x), float(lm.y)) for idx, lm in enumerate(landmarks)}


