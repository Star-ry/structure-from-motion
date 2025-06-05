# Week‑12 Structure‑from‑Motion Pipeline (OpenCV‑free, **Checkerboard**)
# ======================================================================
# Pipeline modules covered
# -----------------------
# 1. Automatic checkerboard corner detection & sub‑pixel refinement
# 2. Automatic correspondence ordering (row‑major grid)
# 3. Two‑view homography estimation with RANSAC (scikit‑image)
# 4. Intrinsics via Zhang’s method
# 5. Extrinsics (R, t) per view
# 6. 3‑D visualisation of camera centres & checkerboard plane
# 7. (Optional hooks left) – bundle adjustment, COLMAP comparison
#
#
# Directory structure
# -------------------
# • data/captured_images/       <-- your JPG/PNG photos
# • data/feature_extraction/    <-- detection overlays written here
# • data/feature_matching/      <-- (kept for rubric, stores RANSAC inlier plots)
#
# ======================================================================

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage import color, feature, transform, measure
from scipy.linalg import svd, inv

# ───────────────────────────────────────────────────────────────────────
# 0. Fixed parameters & directories
# ───────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
images_dir = ROOT / "data/captured_images"          # <‑‑ put photos here
feature_extraction_dir = ROOT / "data/feature_extraction"
feature_matching_dir    = ROOT / "data/feature_matching"

for d in [feature_extraction_dir, feature_matching_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Checkerboard spec – **edit these to match your print‑out** -------------
# CHESS_ROWS & COLS are the number of **internal corners** (not squares!)
CHESS_ROWS  = 7   # ← example: 7 rows of inner corners
CHESS_COLS  = 9   # ← example: 9 columns of inner corners
SQUARE_SIZE_MM = 24.0  # 1 square edge in millimetres (real‑world)

# Derived constants
NUM_CORNERS = CHESS_ROWS * CHESS_COLS

# ───────────────────────────────────────────────────────────────────────
# 1. Feature extraction – checkerboard corner detection
# ───────────────────────────────────────────────────────────────────────

@dataclass
class View:
    name:    str
    rgb:     np.ndarray
    corners: np.ndarray               # (N, 2) pixels ordered row‑major
    H:       np.ndarray = None        # 3×3 homography (world→image)
    R:       np.ndarray = None        # 3×3 rotation
    t:       np.ndarray = None        # 3×1 translation


def detect_checkerboard_corners(img_rgb: np.ndarray,
                                rows: int = CHESS_ROWS,
                                cols: int = CHESS_COLS) -> np.ndarray:
    """Return (rows*cols, 2) array of pixel coords ordered row‑major.
    No OpenCV – uses Harris + peak detection + simple grid ordering.
    """
    gray = color.rgb2gray(img_rgb)
    # Harris corner response -------------------------------------------
    harr_response = feature.corner_harris(gray, k=0.04, sigma=2)
    peaks = feature.corner_peaks(harr_response,
                                 min_distance=5,
                                 threshold_rel=0.02,
                                 num_peaks=rows*cols*4)  # over‑sample

    if len(peaks) < rows*cols:
        raise RuntimeError(f"Only {len(peaks)} peaks found; need {rows*cols}.")

    # Sub‑pixel refinement
    peaks_sub = feature.corner_subpix(gray,
                                      peaks.astype(float),
                                      window_size=11,
                                      alpha=0.1)
    # Remove failed refinements (NaNs)
    peaks_sub = peaks_sub[~np.isnan(peaks_sub).any(axis=1)]

    # Choose the strongest N peaks -------------------------------------
    # Sort by Harris response magnitude at integer location
    responses = harr_response[peaks[:, 0], peaks[:, 1]]
    idx = np.argsort(responses)[::-1][:rows*cols]
    corners = peaks_sub[idx]          # (N, 2) in (row, col) order

    # Order into grid (row‑major) --------------------------------------
    # 1. k‑means‑like clustering on y to determine rows
    ys = corners[:, 0]
    sorted_y_idx = np.argsort(ys)
    corners_sorted = corners[sorted_y_idx]

    # Split into rows by grouping consecutive points so that the y‑gap
    # between adjacent points is small compared to median row height.
    row_indices: List[np.ndarray] = []
    current = [0]
    for i in range(1, len(corners_sorted)):
        if abs(corners_sorted[i, 0] - corners_sorted[i-1, 0]) < 0.5 * np.median(np.diff(ys)):
            current.append(i)
        else:
            row_indices.append(np.array(current))
            current = [i]
    row_indices.append(np.array(current))

    # If we ended up with too many/few rows, fall back to uniform split
    if len(row_indices) != rows:
        corners_sorted = corners_sorted[np.argsort(corners_sorted[:, 0])]
        row_indices = np.array_split(np.arange(rows*cols), rows)

    # Within each row, sort by x (left→right) and collect
    ordered = []
    for r_idx in row_indices:
        row_pts = corners_sorted[r_idx]
        ordered.extend(row_pts[np.argsort(row_pts[:, 1])])

    ordered = np.array([[pt[1], pt[0]] for pt in ordered])  # to (x, y)
    if ordered.shape[0] != rows*cols:
        raise RuntimeError("Grid ordering failed – check parameters.")
    return ordered  # (N, 2)


def visualise_corners(view: View, out_dir: Path):
    fig, ax = plt.subplots()
    ax.imshow(view.rgb)

    xs, ys = view.corners[:, 0], view.corners[:, 1]
    ax.scatter(xs, ys, s=10, c="lime")
    for k, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, str(k), color="yellow", fontsize=6, ha="center", va="center")
    ax.axis("off")

    out = out_dir / f"{view.name}_corners.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Corners visualised: {out.relative_to(ROOT)}")

# ───────────────────────────────────────────────────────────────────────
# 2. Homography estimation (world plane → image)
# ───────────────────────────────────────────────────────────────────────

def estimate_homography_RANSAC(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Returns a 3×3 H such that dst ≃ H * src (homogeneous)."""
    model, inliers = measure.ransac((src_pts, dst_pts),
                                    transform.ProjectiveTransform,
                                    min_samples=4,
                                    residual_threshold=2.0,
                                    max_trials=1000)
    print(f"    · {np.count_nonzero(inliers)}/{len(inliers)} inliers")

    # Visualise inliers & outliers -------------------------------------
    fig, ax = plt.subplots()
    ax.plot(src_pts[inliers, 0], src_pts[inliers, 1], "go", label="Inliers")
    ax.plot(src_pts[~inliers, 0], src_pts[~inliers, 1], "ro", label="Outliers")
    ax.legend()
    ax.set_title("RANSAC inlier mask (world plane)")
    out = feature_matching_dir / f"ransac_{np.random.randint(1e6)}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return model.params

# ───────────────────────────────────────────────────────────────────────
# 3. Zhang intrinsics from multiple homographies
# ───────────────────────────────────────────────────────────────────────

def v_ij(H, i, j):
    return np.array([
        H[0, i]*H[0, j],
        H[0, i]*H[1, j] + H[1, i]*H[0, j],
        H[1, i]*H[1, j],
        H[2, i]*H[0, j] + H[0, i]*H[2, j],
        H[2, i]*H[1, j] + H[1, i]*H[2, j],
        H[2, i]*H[2, j]
    ])

def compute_intrinsics(homographies: List[np.ndarray]) -> np.ndarray:
    V = []
    for H in homographies:
        V.append(v_ij(H, 0, 1))
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
    V = np.stack(V, axis=0)
    _, _, vh = svd(V)
    b = vh[-1]
    v0 = (b[1]*b[3] - b[0]*b[4]) / (b[0]*b[2] - b[1]**2)
    lamb = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4])) / b[0]
    alpha = np.sqrt(lamb / b[0])
    beta  = np.sqrt(lamb * b[0] / (b[0]*b[2] - b[1]**2))
    gamma = -b[1] * alpha**2 * beta / lamb
    u0    = gamma * v0 / beta - b[3] * alpha**2 / lamb
    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,     1]])
    return K

# ───────────────────────────────────────────────────────────────────────
# 4. Extrinsics decomposition per view
# ───────────────────────────────────────────────────────────────────────

def decompose_homography(H: np.ndarray, K: np.ndarray):
    K_inv = inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    t  = lam * (K_inv @ h3)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = svd(R)
    R = U @ Vt  # ensure R is proper rotation
    return R, t.reshape(3, 1)

# ───────────────────────────────────────────────────────────────────────
# 5. 3‑D visualisation
# ───────────────────────────────────────────────────────────────────────

def plot_scene(views: List[View], K: np.ndarray):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Checkerboard plane (z=0)
    plane_w = (CHESS_COLS-1) * SQUARE_SIZE_MM
    plane_h = (CHESS_ROWS-1) * SQUARE_SIZE_MM
    X, Y = np.meshgrid([0, plane_w], [0, plane_h])
    ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.1, color="gray")

    for v in views:
        if v.R is None or v.t is None:
            continue
        C = (-v.R.T @ v.t).flatten()     # (3,) float
        ax.scatter(C[0],C[1],C[2], marker="o")
        ax.text(C[0],C[1],C[2], v.name, fontsize=8)

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Estimated camera centres & checkerboard plane")
    out = ROOT / "data/camera_centres.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Scene visualised: {out.relative_to(ROOT)}")

# ───────────────────────────────────────────────────────────────────────
# 6. Main pipeline
# ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Load images ----------------------------------------------------
    img_paths = sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.png"))
    assert img_paths, "No images found in captured_images/"

    views: List[View] = []
    print("[+] Detecting checkerboard corners …")
    for p in img_paths:
        rgb = imageio.imread(p)
        try:
            corners = detect_checkerboard_corners(rgb)
        except RuntimeError as e:
            print(f"    ‑‑ {p.name}: {e}")
            continue
        v = View(name=p.stem, rgb=rgb, corners=corners)
        views.append(v)
        visualise_corners(v, feature_extraction_dir)

    # 2. Generate world coordinates for internal corners ---------------
    obj_pts = np.zeros((NUM_CORNERS, 3))
    obj_pts[:, :2] = np.stack(np.meshgrid(np.arange(CHESS_COLS),
                                          np.arange(CHESS_ROWS)), -1).reshape(-1, 2)
    obj_pts[:, :2] *= SQUARE_SIZE_MM

    # 3. Homographies ---------------------------------------------------
    print("[+] Estimating homographies with RANSAC …")
    homographies: List[np.ndarray] = []
    for v in views:
        H = estimate_homography_RANSAC(obj_pts[:, :2], v.corners)
        v.H = H
        homographies.append(H)

    # 4. Intrinsics -----------------------------------------------------
    print("[+] Computing intrinsics (Zhang) …")
    K = compute_intrinsics(homographies)
    print("\nEstimated K:\n", K)

    # 5. Extrinsics -----------------------------------------------------
    print("[+] Decomposing homographies into R, t …")
    for v in views:
        v.R, v.t = decompose_homography(v.H, K)

    # 6. Visualise ------------------------------------------------------
    plot_scene(views, K)

    print("[✓] Pipeline finished!")


if __name__ == "__main__":
    main()
