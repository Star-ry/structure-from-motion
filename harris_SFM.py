from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
np.random.seed(42) 

CHESS_ROWS = 7
CHESS_COLS = 9
SQUARE_SIZE = 40.0
ROWS = CHESS_ROWS + 1
COLS = CHESS_COLS + 1


def natural_key(string_):
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', str(string_))]


def load_images(images_dir, resize=(1024, 1024)):
    images = sorted(images_dir.glob("*.jpg"), key=natural_key)
    img_arrays = [np.array(Image.open(img).convert("L").resize(resize), dtype=np.float32) / 255.0
                  for img in tqdm(images, desc="Loading Images")]
    return img_arrays, images


def sort_corners_grid_order(corners, rows, cols):
    corners = np.array(corners)
    ys = corners[:, 1]  # y좌표 기준
    sorted_y_idx = np.argsort(ys)
    corners_sorted = corners[sorted_y_idx]

    row_indices = []
    current = [0]
    median_y_gap = np.median(np.diff(ys))

    for i in range(1, len(corners_sorted)):
        if abs(corners_sorted[i, 1] - corners_sorted[i-1, 1]) < 0.5 * median_y_gap:
            current.append(i)
        else:
            row_indices.append(np.array(current))
            current = [i]
    row_indices.append(np.array(current))

    if len(row_indices) != rows:
        corners_sorted = corners[np.argsort(corners[:, 1])]
        row_indices = np.array_split(np.arange(rows * cols), rows)

    ordered = []
    for r_idx in row_indices:
        row_pts = corners_sorted[r_idx]
        ordered.extend(row_pts[np.argsort(row_pts[:, 0])])  # x 기준 정렬

    return np.array(ordered)  # shape: (N, 2)


def harris_corner_detector(img, window_size=5, k=0.04, threshold=1e-2, nms_size=7):
    dy, dx = np.gradient(img)
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy

    offset = window_size // 2
    corner_response = np.zeros(img.shape)

    for y in range(offset, img.shape[0] - offset):
        for x in range(offset, img.shape[1] - offset):
            window = slice(y - offset, y + offset + 1), slice(x - offset, x + offset + 1)
            Sxx = np.sum(Ixx[window])
            Syy = np.sum(Iyy[window])
            Sxy = np.sum(Ixy[window])

            M = np.array([[Sxx, Sxy], [Sxy, Syy]])
            R = np.linalg.det(M) - k * (np.trace(M) ** 2)
            corner_response[y, x] = R

    # Threshold
    corner_mask = corner_response > threshold
    candidate_coords = np.argwhere(corner_mask)

    # NMS
    half = nms_size // 2
    nms_corners = []
    for y, x in candidate_coords:
        y0, y1 = max(0, y - half), min(img.shape[0], y + half + 1)
        x0, x1 = max(0, x - half), min(img.shape[1], x + half + 1)
        local_patch = corner_response[y0:y1, x0:x1]
        if corner_response[y, x] == np.max(local_patch):
            nms_corners.append((x, y))  # (x, y) 순서

    # Top-N 필터링 (INTERNAL 코너만 유지)
    scores = [corner_response[y, x] for x, y in nms_corners]
    top_n = ROWS * COLS
    if len(nms_corners) > top_n:
        top_indices = np.argsort(scores)[-top_n:]  # 상위 N개 인덱스
        nms_corners = [nms_corners[i] for i in top_indices]

    return nms_corners




def normalize_points(pts):
    pts = np.asarray(pts, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    pts_shifted = pts - centroid
    dists = np.linalg.norm(pts_shifted, axis=1)
    mean_dist = np.mean(dists)

    if mean_dist < 1e-8:
        raise ValueError("All input points are the same or extremely close.")
    
    scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T[:, :2]
    return pts_norm, T

def estimate_homography(matches):
    img_pts = np.array([m[0] for m in matches])       # (u, v)
    world_2d_pts = np.array([m[1][:2] for m in matches])  # (X, Y)

    if len(matches) < 4:
        raise ValueError("At least 4 matches are required to estimate homography.")
    
    img_pts_norm, T_img = normalize_points(img_pts)
    world_pts_norm, T_world = normalize_points(world_2d_pts)

    A = []
    for (u, v), (X, Y) in zip(img_pts_norm, world_pts_norm):
        A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
    
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    # 정규화된 H를 원래 좌표계로 되돌리기
    H = np.linalg.inv(T_img) @ H_norm @ T_world
    if np.linalg.det(H) < 0:
        H = -H

    # # 내부에서 Homography를 적용해보고 결과 확인
    # pts_homog = np.hstack([world_2d_pts, np.ones((len(world_2d_pts), 1))])  # (X, Y, 1)
    # projected = (H @ pts_homog.T).T
    # projected /= projected[:, 2][:, np.newaxis]  # Normalize
    # projected_pts = projected[:, :2]

    # print("=== Homography Matrix H ===")
    # print(np.round(H, 3))
    # print("\n=== Original Image Points ===")
    # print(img_pts)
    # print("\n=== Projected Points (from world coords using H) ===")
    # print(np.round(projected_pts, 2))
    # print("Condition number:", np.linalg.cond(H))
    # print("Det number:", np.linalg.det(H))
    # if np.linalg.det(H)<0:
    #     print()
    # cond_A = S[0] / S[-1]
    # print("Condition number of A matrix:", cond_A)

    # reproj_error = np.linalg.norm(projected_pts - img_pts, axis=1)
    # print("Reprojection error (pixels):", np.round(reproj_error, 3))
    # print("Mean reprojection error:", np.mean(reproj_error))
    return H



def ransac_homography(matches, num_iter=5000, threshold=2.0):
    best_inliers = []
    best_H = None
    matches = np.array(matches, dtype=object)

    for i in range(num_iter):
        indices = np.random.choice(len(matches), 4, replace=False)
        subset = matches[indices]
        H = estimate_homography(subset)

        if not np.all(np.isfinite(H)) or np.linalg.matrix_rank(H) < 3:
            continue  # skip invalid H

        inliers = []
        for match in matches:
            (x1, y1) = match[0] # Image point
            (x2, y2, z2) = match[1] # World point (X, Y, Z) 

            p_world_homo = np.array([x2, y2, 1.0]) 
            p_img_proj_homo = H @ p_world_homo   

            if np.abs(p_img_proj_homo[-1]) < 1e-6: 
                continue
            p_img_proj = p_img_proj_homo[:2] / p_img_proj_homo[-1] 

            error = np.linalg.norm(p_img_proj - np.array([x1, y1]))
            if error < threshold:
                inliers.append(match) 

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

        if i % 1000 == 0:
            tqdm.write(f"Iteration {i}: Best inliers so far = {len(best_inliers)}")

    if best_H is not None and len(best_inliers) >= 40: 
        best_H = estimate_homography(best_inliers)
        det = np.linalg.det(best_H)
        cond = np.linalg.cond(best_H)
        tqdm.write(f"✔️ Final H: det={det:.6f}, cond={cond:.2f}, Inliers: {len(best_inliers)}")
    else:
        print(f"⚠️ RANSAC failed: inliers = {len(best_inliers)} < 40")
        best_H = None

    return best_H, best_inliers



def compute_intrinsics_from_homographies(homographies):
    def v_ij(H, i, j):
        return np.array([
            H[0, i]*H[0, j],
            H[0, i]*H[1, j] + H[1, i]*H[0, j],
            H[1, i]*H[1, j],
            H[2, i]*H[0, j] + H[0, i]*H[2, j],
            H[2, i]*H[1, j] + H[1, i]*H[2, j],
            H[2, i]*H[2, j]
        ])

    V = []
    for H in homographies:
        H = H / H[-1, -1]  # Normalize
        v12 = v_ij(H, 0, 1)
        v11 = v_ij(H, 0, 0)
        v22 = v_ij(H, 1, 1)
        V.append(v12)
        V.append((v11 - v22))

    V = np.array(V)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1] 

    B11, B12, B22, B13, B23, B33 = b

    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = np.sqrt(lamda / B11)
    beta = np.sqrt(lamda * B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lamda
    u0 = gamma * v0 / beta - B13 * alpha**2 / lamda

    # Intrinsic matrix K
    K = np.array([
        [alpha, gamma, u0],
        [0,  beta, v0],
        [0,   0, 1]
    ])

    return K


def compute_extrinsics(H, K):
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    
    # Calculate lambda
    lambda_sq = 1.0 / (np.linalg.norm(K_inv @ h1) * np.linalg.norm(K_inv @ h2))
    lam = np.sqrt(lambda_sq)

    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    R_approx = np.stack([r1, r2, r3], axis=1)

    # 정규 직교화 (rotation matrix 보정)
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    t = lam * (K_inv @ h3)
    return R, t


def plot_camera_views(Rs, ts, image_names=None, grid_size=(CHESS_COLS, CHESS_ROWS), square_size=SQUARE_SIZE):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 1. 체커보드 그리드 평면 그리기 (World Coordinate System)
    grid_x, grid_y = np.meshgrid(
        np.arange(grid_size[0] + 1) * square_size,
        np.arange(grid_size[1] + 1) * square_size
    )
    grid_z = np.zeros_like(grid_x)
    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.2, color='gray', edgecolor='black')

    # 2. 카메라 좌표 수집 및 그리기
    cam_positions = []
    for i, (R, t) in enumerate(zip(Rs, ts)):
        cam_pos = -R.T @ t
        cam_positions.append(cam_pos)
        ax.scatter(*cam_pos, c='r', marker='o', s=50)
        if image_names:
            if isinstance(image_names[i], tuple):
                label = image_names[i][1].stem 
            else:
                label = image_names[i].stem 
            ax.text(*cam_pos, label, fontsize=15, color='blue') 


    # 3. 동일 스케일 설정
    cam_positions = np.array(cam_positions)
    all_x = np.concatenate([grid_x.flatten(), cam_positions[:, 0]])
    all_y = np.concatenate([grid_y.flatten(), cam_positions[:, 1]])
    all_z = np.concatenate([grid_z.flatten(), cam_positions[:, 2]])

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()
    max_range = max(x_range, y_range, z_range)

    mid_x = (all_x.max() + all_x.min()) / 2
    mid_y = (all_y.max() + all_y.min()) / 2
    mid_z = (all_z.max() + all_z.min()) / 2

    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)


    ax.set_xlabel("X (mm)") # Added units
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Estimated Camera Poses and Chessboard Plane")
    ax.view_init(elev=-90, azim=-90)
    plt.tight_layout()
    plt.savefig("CameraPoses.jpg", dpi=300)
    plt.show()

def draw_corners_on_image(img, corners, save_path, draw_index=False):
    img_draw = (img * 255).astype(np.uint8)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
    for idx, (x, y) in enumerate(corners):
        cv2.circle(img_draw, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)
        if draw_index:
            cv2.putText(img_draw, str(idx), (int(x) + 3, int(y) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(save_path), img_draw)



def is_valid_homography(H):
    if H is None or not np.all(np.isfinite(H)):
        return False

    H = H / H[2, 2]  # normalize

    cond = np.linalg.cond(H)
    scale = np.linalg.norm(H)
    det = np.linalg.det(H)

    return scale < 1e5 and 1e-6 < abs(det) < 1e6 and cond < 1000000 # Increased thresholds for robustness


def save_corners_to_txt(corners, path):
    with open(path, 'w') as f:
        for x, y in corners:
            f.write(f"{x:.4f} {y:.4f}\n")


def load_corners_from_txt(path):
    corners = []
    with open(path, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split())
            corners.append((x, y))
    return corners


def visualize_homography_projection(img, detected_corners, world_points_2d, H, save_path=None):
    """
    Visualizes the detected 2D corners on the image and their projection
    from the world points using the estimated Homography.
    """
    img_vis = (img * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    # Project world points into the image using H
    projected_points = []
    for (x_w, y_w) in world_points_2d:
        p_w = np.array([x_w, y_w, 1.0])
        p_img_proj = H @ p_w
        if abs(p_img_proj[2]) < 1e-6: # Avoid division by zero
            projected_points.append(None)
            continue
        p_img_proj /= p_img_proj[2]
        projected_points.append((int(p_img_proj[0]), int(p_img_proj[1])))

    for idx, (detected_x, detected_y) in enumerate(detected_corners):
        # Draw detected corner (from image)
        cv2.circle(img_color, (int(detected_x), int(detected_y)), 5, (0, 255, 0), -1) # Green for detected

        # Draw projected corner (from world points via H)
        proj_pt = projected_points[idx]
        if proj_pt:
            cv2.circle(img_color, proj_pt, 5, (0, 0, 255), -1) # Red for projected
            # Draw line between detected and projected (reprojection error visualization)
            cv2.line(img_color, (int(detected_x), int(detected_y)), proj_pt, (255, 0, 0), 1) # Blue line

        # Add index text
        cv2.putText(img_color, str(idx), (int(detected_x) + 7, int(detected_y) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if save_path:
        cv2.imwrite(str(save_path), img_color)
    else:
        cv2.imshow("Homography Projection", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




def create_world_object_points(rows, cols, square_size):
    """
    Generates the 3D coordinates of the chessboard corners in the world coordinate system.
    The origin (0,0,0) is typically at the top-left internal corner.
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def draw_world_points(world_points, scale=1.0, margin=50, save_path="world_points_visualization.jpg"):
    """
    Draws 2D world points (X, Y) on a black background with index labels.

    Parameters:
    - world_points: list of (X, Y, Z) or (X, Y)
    - scale: scale factor to enlarge spacing for visualization
    - margin: extra space around points
    - save_path: where to save the visualization
    """
    pts_2d = np.array([pt[:2] for pt in world_points], dtype=np.float32)
    pts_scaled = pts_2d * scale

    min_xy = pts_scaled.min(axis=0)
    pts_shifted = pts_scaled - min_xy + margin

    max_xy = pts_shifted.max(axis=0).astype(int) + margin
    img_size = (int(max_xy[1]), int(max_xy[0]), 3)  # H, W, C
    canvas = np.zeros(img_size, dtype=np.uint8)

    for i, (x, y) in enumerate(pts_shifted.astype(int)):
        cv2.circle(canvas, (x, y), radius=5, color=(255, 255, 255), thickness=-1)
        cv2.putText(canvas, str(i), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)
    print(f"✅ Saved world point visualization to {save_path}")


def main(data_dir, result_dir):
    images_dir = Path(data_dir) / "images"
    feature_extraction_dir = Path(result_dir) / "feature_extraction"
    feature_matching_dir = Path(result_dir) / "feature_matching"
    feature_extraction_dir.mkdir(parents=True, exist_ok=True)
    feature_matching_dir.mkdir(parents=True, exist_ok=True)

    imgs, image_paths = load_images(images_dir)
    print("Images Loaded")

    # Create World Object Points
    world_object_points = create_world_object_points(ROWS, COLS, SQUARE_SIZE)
    print(f"Generated {len(world_object_points)} world object points.")


    # Feature Extraction
    corners_list = [] 
    for img, path in tqdm(zip(imgs, image_paths), desc="Corner Detection", total=len(imgs)):
        img_idx = path.stem.split('_')[-1]
        corners_txt_path = feature_extraction_dir / f"corners_{img_idx}.txt"
        corners_img_path = feature_extraction_dir / f"corners_{img_idx}.jpg"

        if corners_txt_path.exists():
            corners = load_corners_from_txt(corners_txt_path)
            corners = np.array(corners)
        else:
            corners = harris_corner_detector(img)
            corners = np.array(corners)
            save_corners_to_txt(corners, corners_txt_path)

        if len(corners) == len(world_object_points):
            corners = sort_corners_grid_order(corners, ROWS, COLS)
            draw_corners_on_image(img, corners, corners_img_path, draw_index=True)
            corners_list.append(corners)
        else:
            tqdm.write(f"Skipping image {path.stem}: Detected {len(corners)} corners, expected {len(world_object_points)}. Try adjusting Harris parameters or image quality.")
            corners_list.append(None) # Append None to maintain list alignment, will be filtered later

    print("Feature Extracted")


    # Homography Estimation (3D world points to 2D image points)
    homographies = []
    corresponding_image_paths_for_H = []
    all_matches_for_vis = []
    draw_world_points(world_object_points.tolist(), scale=1.0, save_path="world_points_grid.jpg")

    for idx, (img, corners, img_path) in enumerate(zip(imgs, corners_list, image_paths)):
        matches = list(zip(corners.tolist(), world_object_points.tolist()))

        if len(matches) >= 4:
            tqdm.write(f"\n[{idx+1}/{len(imgs)}] Estimating Homography for {img_path.stem}...")
            H, inliers = ransac_homography(matches)

            if H is None:
                continue # Skip if RANSAC failed

            homographies.append(H)
            corresponding_image_paths_for_H.append(img_path)
            all_matches_for_vis.append(matches)

            visualize_homography_projection(img, corners, world_object_points[:, :2], H,
                                            save_path=feature_matching_dir / f"[{idx}: {len(inliers)}] {img_path.stem}.jpg")


    if not homographies:
        print("No valid homographies could be estimated. Exiting.")
        return



    # Estimate Intrinsics 
    filtered_homographies_for_intrinsics = [H for H in homographies if is_valid_homography(H)]
    print("\nFiltered Homographies ===================================\n")
    for i, H in enumerate(filtered_homographies_for_intrinsics):
        det = np.linalg.det(H)
        cond = np.linalg.cond(H)
        print(f"[{corresponding_image_paths_for_H[i].stem}] Determinant: {det:.6f}, Condition number: {cond:.2f}")

    K = compute_intrinsics_from_homographies(filtered_homographies_for_intrinsics)
    if K is None:
        print("Failed to compute intrinsic matrix (K). Exiting.")
        return
    print("\nEstimated Intrinsics (K):\n", K)


    # Estimate  Extrinsics
    extrinsics = []
    for H in tqdm(homographies, desc="Computing Extrinsics"):
        R, t = compute_extrinsics(H, K)
        extrinsics.append((R, t))

    Rs, ts = zip(*extrinsics)
    plot_camera_views(Rs, ts, image_names=corresponding_image_paths_for_H)


if __name__ == "__main__":
    data_dir = "/home/starry/workspace/works/threed_computer_vision/icpbl/data"
    result_dir = "/home/starry/workspace/works/threed_computer_vision/icpbl/results"
    main(data_dir, result_dir)