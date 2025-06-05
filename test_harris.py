from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


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


# def sort_corners_grid_order(corners, rows, cols):
#     """
#     Sorts corners strictly in row-major order: top-left to bottom-right.
#     Ensures each row has `cols` number of points sorted by x.
#     """
#     corners = np.array(corners)
#     sorted_by_y = corners[np.argsort(corners[:, 1])]
#     row_list = []

#     # Split into rows using y-value windowing
#     y_coords = sorted_by_y[:, 1]
#     row_indices = np.linspace(0, len(y_coords), rows + 1, dtype=int)

#     for i in range(rows):
#         row = sorted_by_y[row_indices[i]:row_indices[i + 1]]
#         row_sorted = row[np.argsort(row[:, 0])]
#         row_list.append(row_sorted)

#     sorted_corners = np.vstack(row_list)
#     if sorted_corners.shape[0] != rows * cols:
#         print(f"[Warning] Expected {rows * cols} corners but got {sorted_corners.shape[0]}")
    
#     return [tuple(pt) for pt in sorted_corners]

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

    # corners = fit_homography_grid(nms_corners, INTERNAL_ROWS, INTERNAL_COLS)

    # # corner 그리드 순서대로 변환
    # corners = sort_corners_grid_order(nms_corners, ROWS, COLS)

    return nms_corners


def extract_patch(img, pt, window_size):
    x, y = map(int, pt)  # float -> int 변환
    half = window_size // 2
    if y - half < 0 or y + half >= img.shape[0] or x - half < 0 or x + half >= img.shape[1]:
        return None
    return img[y - half:y + half + 1, x - half:x + half + 1]


def match_features(img1, corners1, img2, corners2, window_size=5):
    if len(corners1) != len(corners2):
        print(f"[Warning] Corner count mismatch: {len(corners1)} vs {len(corners2)}")
        return []

    return list(zip(corners1, corners2))


def normalize_points(pts):
    pts = np.array(pts)
    centroid = np.mean(pts, axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(pts - centroid, axis=1))
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0,     0,               1]])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def estimate_homography(matches):
    src_pts = [m[0] for m in matches]
    dst_pts = [m[1] for m in matches]

    src_norm, T1 = normalize_points(src_pts)
    dst_norm, T2 = normalize_points(dst_pts)

    A = []
    for (x1, y1), (x2, y2) in zip(src_norm, dst_norm):
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H_norm = V[-1].reshape(3, 3)

    H = np.linalg.inv(T2) @ H_norm @ T1
    return H / H[2, 2]

def ransac_homography(matches, num_iter=100, threshold=1.0):
    best_inliers = []
    best_H = None
    matches = np.array(matches)

    for _ in range(num_iter):
        subset = matches[np.random.choice(len(matches), 4, replace=False)]
        H = estimate_homography(subset)

        if not np.all(np.isfinite(H)) or np.linalg.matrix_rank(H) < 3:
            continue  # skip invalid H

        inliers = []
        for (x1, y1), (x2, y2) in matches:
            p1 = np.array([x1, y1, 1.0])
            p2_est = H @ p1
            if abs(p2_est[2]) < 1e-6:
                continue  # skip unstable projection
            p2_est /= p2_est[2]
            error = np.linalg.norm(p2_est[:2] - np.array([x2, y2]))
            if error < threshold:
                inliers.append(((x1, y1), (x2, y2)))

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    if best_H is not None and len(best_inliers) >= 10:
        best_H = estimate_homography(best_inliers)

    return best_H, best_inliers


def compute_intrinsics_from_homographies(homographies):
    def compute_v(H, i, j):
        return np.array([
            H[0, i]*H[0, j],
            H[0, i]*H[1, j] + H[1, i]*H[0, j],
            H[1, i]*H[1, j],
            H[2, i]*H[0, j] + H[0, i]*H[2, j],
            H[2, i]*H[1, j] + H[1, i]*H[2, j],
            H[2, i]*H[2, j]
        ])

    if len(homographies) < 3:
        print("[Error] Not enough homographies for intrinsic estimation.")
        return np.eye(3)

    V = []
    for H in homographies:
        V.append(compute_v(H, 0, 1))
        V.append(compute_v(H, 0, 0) - compute_v(H, 1, 1))
    V = np.array(V)
    _, _, Vh = np.linalg.svd(V)
    b = Vh[-1]

    denom = b[0]*b[2] - b[1]**2
    if denom == 0 or b[0] == 0:
        print("[Error] Invalid denominator in intrinsic computation.")
        return np.eye(3)

    v0 = (b[1]*b[3] - b[0]*b[4]) / denom
    lam = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4])) / b[0]

    # 수치적 안정성 체크
    if lam <= 0 or (lam / b[0]) <= 0 or (lam * b[0] / denom) <= 0:
        print("[Error] Invalid lambda or sqrt argument.")
        return np.eye(3)

    alpha = np.sqrt(lam / b[0])
    beta = np.sqrt(lam * b[0] / denom)
    gamma = -b[1] * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - b[3] * alpha**2 / lam

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,     1]])
    return K


def compute_extrinsics(H, K):
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    R_approx = np.stack([r1, r2, r3], axis=1)

    # 정규 직교화 (rotation matrix 보정)
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    t = lam * (K_inv @ h3)
    return R, t


def plot_camera_views(Rs, ts, image_names=None, grid_size=(COLS, ROWS), square_size=SQUARE_SIZE):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 1. 체커보드 그리드 평면 그리기
    grid_x, grid_y = np.meshgrid(
        np.arange(grid_size[0] + 1) * square_size,
        np.arange(grid_size[1] + 1) * square_size
    )
    grid_z = np.zeros_like(grid_x)
    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.2, color='gray', edgecolor='black')

    # 2. 카메라 좌표 수집
    cam_positions = []
    for i, (R, t) in enumerate(zip(Rs, ts)):
        cam_pos = -R.T @ t
        cam_positions.append(cam_pos)
        ax.scatter(*cam_pos, c='r')
        if image_names:
            label = image_names[i].stem
            ax.text(*cam_pos, label, fontsize=12, color='black')

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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Estimated Camera Poses with Grid Plane")
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
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

def draw_feature_matches(img1, corners1, img2, corners2, matches, save_path):
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    height = max(img1_color.shape[0], img2_color.shape[0])
    combined = np.zeros((height, img1_color.shape[1] + img2_color.shape[1], 3), dtype=np.uint8)
    combined[:img1_color.shape[0], :img1_color.shape[1]] = img1_color
    combined[:img2_color.shape[0], img1_color.shape[1]:] = img2_color

    for idx, ((x1, y1), (x2, y2)) in enumerate(matches):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + img1_color.shape[1], int(y2))
        cv2.line(combined, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(combined, pt1, 2, (0, 0, 255), -1)
        cv2.circle(combined, pt2, 2, (0, 0, 255), -1)

        cv2.putText(combined, str(idx), (pt1[0] + 3, pt1[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
        cv2.putText(combined, str(idx), (pt2[0] + 3, pt2[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

    cv2.imwrite(str(save_path), combined)

def is_valid_homography(H, max_norm=1000, max_cond=1e7):
    return (
        np.all(np.isfinite(H)) and
        np.linalg.norm(H) < max_norm and
        np.linalg.matrix_rank(H) == 3 and
        np.linalg.cond(H) < max_cond
    )


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

def match_features_ssd(img1, corners1, img2, corners2, window_size=11):
    matches = []
    half = window_size // 2

    for i, pt1 in enumerate(corners1):
        patch1 = extract_patch(img1, pt1, window_size)
        if patch1 is None:
            continue

        best_match = None
        best_score = float('inf')

        for j, pt2 in enumerate(corners2):
            patch2 = extract_patch(img2, pt2, window_size)
            if patch2 is None:
                continue

            # SSD (sum of squared differences)
            score = np.sum((patch1 - patch2) ** 2)
            if score < best_score:
                best_score = score
                best_match = (pt1, pt2)

        if best_match is not None:
            matches.append(best_match)

    return matches


def main(data_dir):
    images_dir = Path(data_dir) / "images"
    feature_extraction_dir = Path(data_dir) / "feature_extraction"
    feature_matching_dir = Path(data_dir) / "feature_matching"
    feature_extraction_dir.mkdir(parents=True, exist_ok=True)
    feature_matching_dir.mkdir(parents=True, exist_ok=True)

    imgs, image_paths = load_images(images_dir)
    print("Images Loaded")


    # Feature Extraction
    corners_list = []
    for img, path in tqdm(zip(imgs, image_paths), desc="Corner Detection", total=len(imgs)):
        img_idx = path.stem.split('_')[-1]
        corners_txt_path = feature_extraction_dir / f"corners_{img_idx}.txt"
        corners_img_path = feature_extraction_dir / f"corners_{img_idx}.jpg"

        if corners_txt_path.exists():
            corners = load_corners_from_txt(corners_txt_path)
        else:
            corners = harris_corner_detector(img)
            save_corners_to_txt(corners, corners_txt_path)
        # corner 그리드 순서대로 변환
        corners = sort_corners_grid_order(corners, ROWS, COLS)
        draw_corners_on_image(img, corners, corners_img_path, draw_index=True)

        corners_list.append(corners)
    print("Feature Extracted")


    # Feature Matching
    matches_list = [match_features_ssd(imgs[i], corners_list[i], imgs[i+1], corners_list[i+1])
                    for i in tqdm(range(len(imgs) - 1), desc="Feature Matching")]
    for i, matches in tqdm(list(enumerate(matches_list)), desc="Estimating Homographies and Saving Matches"):
        img1_name = image_paths[i].stem
        img2_name = image_paths[i + 1].stem
        save_path = feature_matching_dir / f"{img1_name}_vs_{img2_name}.jpg"
        draw_feature_matches(imgs[i], corners_list[i], imgs[i + 1], corners_list[i + 1], matches, save_path)
    print("Feature Matched")


    # Estimate Homography
    homographies = []
    used_image_names = []
    for i, matches in tqdm(list(enumerate(matches_list)), desc="Estimating Homographies with RANSAC"):
        if len(matches) >= 4:
            H, inliers = ransac_homography(matches)
            if H is not None and is_valid_homography(H):
                condition = np.linalg.cond(H)
                rank = np.linalg.matrix_rank(H)
                if rank == 3 and condition < 1e8:
                    homographies.append(H)
                    used_image_names.append(image_paths[i])
                else:
                    print(f"[Skipped] H[{i}] has bad condition={condition:.1e}, rank={rank}")
            else:
                print(f"[Invalid] H[{i}] is not valid")
        else:
            print(f"[Warning] Not enough matches between image {i} and {i+1}")
    
    homographies = [H for H in homographies if is_valid_homography(H)]
    print("Estimated Homography")


    # Estimate Intrinsics & Extrinsics
    if homographies:
        K = compute_intrinsics_from_homographies(homographies)
        print("Estimated Intrinsics (K):\n", K)

        extrinsics = [compute_extrinsics(H, K) for H in tqdm(homographies, desc="Computing Extrinsics")]
        Rs, ts = zip(*extrinsics)
        plot_camera_views(Rs, ts, image_names=used_image_names)


if __name__ == "__main__":
    data_dir = "/home/starry/workspace/works/threed_computer_vision/icpbl/data"
    main(data_dir)
