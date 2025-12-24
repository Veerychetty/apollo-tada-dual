import os
import uuid
import cv2
import numpy as np
import json
import traceback
from collections import OrderedDict
import re
import heapq
import shutil 
import time
import logging
from flask import Flask, request, jsonify, send_file,Response
from flask_cors import CORS
import boto3
from botocore.config import Config
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from numba import njit
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI Agg backend for image rendering only
import matplotlib.pyplot as plt
from dotenv import load_dotenv


# --- ENVIRONMENT & FLASK SETUP ---



load_dotenv("file.env")  # Automatically loads all vars in file.env

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

_default_origins = os.environ.get(
    "FRONTEND_ORIGINS",
    ",".join([
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ])
)
ALLOWED_ORIGINS = [origin.strip() for origin in _default_origins.split(',') if origin.strip()]

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS or ["*"]}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Origin", "Accept"],
    methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"]
)

app.logger.setLevel(logging.DEBUG)


# --- Timing Utilities ---


def start_timer():
    return time.perf_counter()


def log_duration(step_name, start_time, details=None):
    elapsed = time.perf_counter() - start_time
    message = f"{step_name} took {elapsed:.2f}s"
    if details:
        message = f"{message} ({details})"
    app.logger.debug(message)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and ("*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS):
        response.headers["Access-Control-Allow-Origin"] = origin
    elif ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS[0]
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Requested-With,Origin,Accept"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS,PUT,DELETE"
    return response

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

# --- Security: Path Validation Functions ---

def sanitize_session_id(session_id):
    """
    Sanitize session ID to prevent path traversal attacks.
    Only allows alphanumeric characters, hyphens, and underscores.
    """
    if not session_id:
        return None
    # Remove any path traversal attempts
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', str(session_id))
    # Ensure it's not empty and not too long
    if not sanitized or len(sanitized) > 100:
        return None
    return sanitized

def validate_path_within_directory(file_path, allowed_directory):
    """
    Validate that a file path is within the allowed directory to prevent path traversal.
    Returns the normalized absolute path if valid, None otherwise.
    """
    try:
        # Get absolute paths
        allowed_abs = os.path.abspath(allowed_directory)
        file_abs = os.path.abspath(file_path)
        
        # Normalize paths to handle .. and . correctly
        allowed_abs = os.path.normpath(allowed_abs)
        file_abs = os.path.normpath(file_abs)
        
        # Check if file path is within allowed directory
        if not file_abs.startswith(allowed_abs):
            app.logger.warning(f"Path traversal attempt detected: {file_path}")
            return None
        
        return file_abs
    except Exception as e:
        app.logger.error(f"Path validation error: {e}")
        return None

def secure_file_path(base_dir, filename):
    """
    Create a secure file path by sanitizing the filename and validating it stays within base_dir.
    """
    # Sanitize filename
    safe_filename = secure_filename(filename)
    if not safe_filename:
        return None
    
    # Join paths
    full_path = os.path.join(base_dir, safe_filename)
    
    # Validate path is within base directory
    validated_path = validate_path_within_directory(full_path, base_dir)
    return validated_path

# AWS Textract Session
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
textract = session.client("textract", config=Config(signature_version="v4"))

# Verify AWS credentials and Textract access
def verify_aws_setup():
    """Verify AWS credentials and Textract service availability"""
    try:
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            app.logger.error("AWS credentials not found in environment variables")
            return False
        
        # Try a simple test to verify credentials and service access
        # Create a minimal test image (1x1 pixel PNG)
        import io
        test_image = io.BytesIO()
        img = Image.new('RGB', (1, 1), color='white')
        img.save(test_image, format='PNG')
        test_image.seek(0)
        
        try:
            response = textract.detect_document_text(Document={"Bytes": test_image.read()})
            app.logger.info("AWS Textract connection verified successfully")
            return True
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            app.logger.error(f"AWS Textract test failed: {error_type}: {error_msg}")
            
            # Check for common error types
            if "UnrecognizedClientException" in error_type or "security token" in error_msg.lower() or "invalid" in error_msg.lower():
                app.logger.error("AWS credentials are EXPIRED or INVALID - please update your credentials in file.env")
                app.logger.error("To fix: 1) Go to AWS IAM Console, 2) Create new Access Keys, 3) Update file.env with new credentials")
            elif "InvalidSignatureException" in error_type or "SignatureDoesNotMatch" in error_msg:
                app.logger.error("AWS credentials appear to be invalid or incorrectly formatted")
            elif "AccessDeniedException" in error_type or "AccessDenied" in error_msg:
                app.logger.error("AWS credentials lack Textract permissions")
            elif "SubscriptionException" in error_type or "subscription" in error_msg.lower():
                app.logger.error("AWS Textract subscription issue detected")
            elif "Region" in error_msg or "region" in error_msg.lower():
                app.logger.error(f"AWS region issue: {AWS_REGION}")
            
            return False
    except Exception as e:
        app.logger.error(f"Failed to verify AWS setup: {type(e).__name__}: {str(e)}")
        return False

# Run verification at startup
if __name__ != "__main__" or True:  # Always verify
    verify_aws_setup()

Image.MAX_IMAGE_PIXELS = None  # Allow large images

SESSIONSDIR = "sessions"
os.makedirs(SESSIONSDIR, exist_ok=True)

ip_storage = {}
MAX_UPLOADS = 20
COOLDOWN_PERIOD = 5 * 60  # 5 minutes in seconds
IPSTORAGEFILE = os.path.join(SESSIONSDIR, "ipstorage.json")
...

# --- Image Dewarping ---

def robust_adaptive_threshold(gray):
    tried = []
    for method in [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]:
        for block_size in [21, 31, 41]:
            for C in [5, 10, 15]:
                try:
                    mask = cv2.adaptiveThreshold(
                        gray, 255, method, cv2.THRESH_BINARY_INV,
                        block_size, C
                    )
                    tried.append(mask)
                except Exception:
                    continue
    for mask in tried:
        if np.count_nonzero(mask) > 0.01 * mask.size:
            return mask
    return tried[-1] if tried else gray

def detect_circle_by_horizontal_scan(image_or_gray, threshold=180, row_step=10, dilation_iter=6):
    if isinstance(image_or_gray, str):
        gray = cv2.imread(image_or_gray, cv2.IMREAD_GRAYSCALE)
    else:
        gray = image_or_gray
    adaptive_thresh = robust_adaptive_threshold(gray)
    kernel = np.ones((4, 4), np.uint8)
    dilated_mask = cv2.dilate(adaptive_thresh, kernel, iterations=dilation_iter)
    h, w = dilated_mask.shape
    points_left, points_right = [], []
    for y in range(0, h, row_step):
        row = dilated_mask[y]
        for x in range(1, w):
            if row[x - 1] > 200 and row[x] < 100:
                points_left.append((x, y))
                break
        for x in range(w-2, 0, -1):
            if row[x + 1] > 200 and row[x] < 100:
                points_right.append((x, y))
                break
    all_points = np.array(points_left + points_right)
    if len(all_points) < 10:
        return None
    x, y = all_points[:, 0], all_points[:, 1]
    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x**2 + y**2
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    xc, yc = sol[0], sol[1]
    radius = np.sqrt(sol[2] + xc**2 + yc**2)
    return [(int(xc), int(yc), int(radius))]

def extract_radial_slice(img, cx, cy, radius, angle):
    angle_rad = np.deg2rad(angle)
    end_x = int(cx + radius * np.cos(angle_rad))
    end_y = int(cy + radius * np.sin(angle_rad))
    line = np.linspace([cx, cy], [end_x, end_y], radius).astype(np.int32)
    h, w = img.shape[:2]
    mask = (line[:, 1] >= 0) & (line[:, 1] < h) & (line[:, 0] >= 0) & (line[:, 0] < w)
    line = line[mask]
    return img[line[:, 1], line[:, 0]]

def count_white_pixels_along_line(img, cx, cy, radius, angle):
    angle_rad = np.deg2rad(angle)
    end_x = int(cx + radius * np.cos(angle_rad))
    end_y = int(cy + radius * np.sin(angle_rad))
    line = np.linspace([cx, cy], [end_x, end_y], radius).astype(np.int32)
    h, w = img.shape
    mask = (line[:, 1] >= 0) & (line[:, 1] < h) & (line[:, 0] >= 0) & (line[:, 0] < w)
    line = line[mask]
    pixels = img[line[:, 1], line[:, 0]]
    return np.sum(pixels == 255)

def find_best_start_angle(img, cx, cy, radius, image_for_overlay=None):
    angles = np.arange(0, 360, 1)
    white_counts = [count_white_pixels_along_line(img, cx, cy, radius, a) for a in angles]
    min_index = np.argmin(white_counts)
    best_angle = angles[min_index]
    if image_for_overlay is not None:
        color_img = image_for_overlay.copy()
        extended_length = int(radius * 1.5)
        angle_rad = np.deg2rad(best_angle)
        x_end = int(cx + extended_length * np.cos(angle_rad))
        y_end = int(cy + extended_length * np.sin(angle_rad))
        cv2.line(color_img, (cx, cy), (x_end, y_end), (0, 0, 255), 3)
        cv2.circle(color_img, (cx, cy), 8, (0, 255, 0), -1)
        cv2.imwrite("best_angle_overlay.jpg", color_img)
    return best_angle

def crop_bottom_silence(gray, margin_min=2500):
    """
    For light images: crop most continuous near-empty rows from bottom up,
    but always leave at least margin_min rows at the top.
    """
    h, w = gray.shape
    threshold = np.percentile(gray, 95) - 12  # aggressive for faint labels
    # start from bottom, count how many rows are 'mostly empty'
    crop_row = h
    for y in range(h-1, margin_min, -1):
        if np.mean(gray[y,:]) > threshold:
            crop_row = y
        else:
            break
    return crop_row

def dewarp_circle_to_rect(img, circle, binary_for_angle, output_path, margin=2200, stretch=3):
    timer = start_timer()
    cx, cy, radius = circle
    best_angle = find_best_start_angle(binary_for_angle, cx, cy, radius, image_for_overlay=img)
    circumference = int(2 * np.pi * radius)
    dewarped = np.zeros((radius, circumference, img.shape[2]), dtype=img.dtype)

    for i in range(circumference):
        angle = (i * (360 / circumference) + best_angle) % 360
        radial_slice = extract_radial_slice(img, cx, cy, radius, angle)
        if radial_slice.shape[0] == radius:
            dewarped[:, i] = radial_slice
        else:
            dewarped[:radial_slice.shape[0], i] = radial_slice

    gray_dewarped = cv2.cvtColor(dewarped, cv2.COLOR_BGR2GRAY)
    mean_gray = np.mean(gray_dewarped)

    if mean_gray > 180:
        # Light/faint: crop just above main label—less blank margin
        use_margin = min(margin, dewarped.shape[0] // 2)  # Typically 2000-2400, can tune
        cropped = dewarped[-use_margin:, :] if dewarped.shape[0] > use_margin else dewarped
    else:
        # Tighter adaptive crop for thick images (both axes)
        crop_threshold = int(np.percentile(gray_dewarped, 10)) + 18
        row_fg = np.where((gray_dewarped < crop_threshold).sum(axis=1) > int(gray_dewarped.shape[1] * 0.01))[0]
        if len(row_fg) > 0:
            top = row_fg[0]
            bottom = row_fg[-1] + 1
            cropped = dewarped[top:bottom, :]
        else:
            cropped = dewarped

        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        col_fg = np.where((gray_cropped < crop_threshold).sum(axis=0) > int(gray_cropped.shape[0] * 0.01))[0]
        if len(col_fg) > 0:
            left = col_fg[0]
            right = col_fg[-1] + 1
            cropped = cropped[:, left:right]

    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        cropped = dewarped

    stretched = cv2.resize(
        cropped, (cropped.shape[1], cropped.shape[0] * stretch),
        interpolation=cv2.INTER_CUBIC
    )
    inverted = cv2.flip(stretched, 0)
    cv2.imwrite(output_path, inverted)
    log_duration("dewarp_circle_to_rect", timer, details=f"radius={radius}")
    return inverted










# --- Chunking with Dijkstra ---

@njit
def dijkstra_numba(mask, start_row, start_col, end_row, penalty_black=2000):
    H, W = mask.shape
    INF = 10**15
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dist = np.full((H, W), INF, dtype=np.int64)
    dist[start_row, start_col] = 0
    visited = np.zeros((H, W), dtype=np.bool_)
    parent_y = np.full((H, W), -1, dtype=np.int32)
    parent_x = np.full((H, W), -1, dtype=np.int32)
    heap = [ (0, start_row, start_col) ]
    while heap:
        cost_u, uy, ux = heapq.heappop(heap)
        if visited[uy, ux]:
            continue
        visited[uy, ux] = True
        if uy == end_row:
            # Reconstruct path
            path_y, path_x = [], []
            cy, cx = uy, ux
            while cy != -1 and cx != -1:
                path_y.append(cy)
                path_x.append(cx)
                py = parent_y[cy, cx]
                px = parent_x[cy, cx]
                cy, cx = py, px
            path = [(path_y[i], path_x[i]) for i in range(len(path_y)-1, -1, -1)]
            return path
        for dy, dx in directions:
            vy, vx = uy + dy, ux + dx
            if 0 <= vy < H and 0 <= vx < W and not visited[vy, vx]:
                penalty = penalty_black if mask[vy, vx] > 0 else 1
                step_penalty = 5 if dx != 0 else 0
                nd = cost_u + penalty + step_penalty
                if nd < dist[vy, vx]:
                    dist[vy, vx] = nd
                    parent_y[vy, vx] = uy
                    parent_x[vy, vx] = ux
                    heapq.heappush(heap, (nd, vy, vx))
    return [(-1, -1)][:0]

def execute_dijkstra_parallel(mask, xs, penalty_black=1000):
    timer = start_timer()
    H, W = mask.shape
    def compute_path(x):
        return dijkstra_numba(mask, 0, x, H-1, penalty_black)
    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(compute_path, xs))
    log_duration("execute_dijkstra_parallel", timer, details=f"paths={len(paths)}")
    return paths

def find_text_density(mask, window_size=40):
    h, w = mask.shape
    density = np.zeros(w)
    for x in range(w):
        density[x] = np.sum(mask[:, x] > 0) / h
    kernel = np.ones(window_size) / window_size
    if len(density) >= window_size:
        density = np.convolve(density, kernel, mode='same')
    return density

def find_optimal_split_points(mask, min_distance=50, max_paths=15):
    h, w = mask.shape
    density = find_text_density(mask)
    valleys = []
    threshold = np.mean(density) * 0.3
    for x in range(min_distance, w-min_distance):
        if density[x] < threshold:
            valleys.append((x, density[x]))
    if not valleys:
        step = w // (max_paths + 1)
        return [(i + 1) * step for i in range(max_paths)]
    valleys.sort(key=lambda x: x[1])
    selected_points = []
    for x, _ in valleys:
        if all(abs(x - existing) >= min_distance for existing in selected_points):
            selected_points.append(x)
            if len(selected_points) >= max_paths:
                break
    if len(selected_points) < max_paths:
        selected_points.sort()
        for i in range(1, max_paths+1):
            uniform_x = i * w // (max_paths + 1)
            if all(abs(uniform_x - existing) >= min_distance for existing in selected_points):
                selected_points.append(uniform_x)
                if len(selected_points) >= max_paths:
                    break
    return sorted(selected_points)

def chunk_dewarped_image(image_path, chunk_folder):
    timer = start_timer()
    
    # Validate image_path to prevent path traversal
    if not image_path:
        raise ValueError("Empty image_path provided")
    
    # Validate image_path is within SESSIONS_DIR
    validated_image_path = validate_path_within_directory(image_path, SESSIONS_DIR)
    if not validated_image_path:
        raise ValueError(f"Invalid image_path: {image_path}")
    
    # Validate chunk_folder is within SESSIONS_DIR
    validated_chunk_folder = validate_path_within_directory(chunk_folder, SESSIONS_DIR)
    if not validated_chunk_folder:
        raise ValueError(f"Invalid chunk_folder: {chunk_folder}")
    
    os.makedirs(validated_chunk_folder, exist_ok=True)
    image = cv2.imread(validated_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read chunking image {validated_image_path}")
    H, W = image.shape
    scale_factor = 0.5
    small_image = cv2.resize(image, (int(W * scale_factor), int(H * scale_factor)))
    adaptive_thresh = cv2.adaptiveThreshold(
        small_image, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        31, 15
    )
    dilated_mask = cv2.dilate(adaptive_thresh, np.ones((15, 15), np.uint8), iterations=9)
    min_split_distance = max(30, dilated_mask.shape[1] // 25)
    optimal_split_points = find_optimal_split_points(dilated_mask, min_split_distance, 15)
    paths = execute_dijkstra_parallel(dilated_mask, optimal_split_points)

    chunk_metadata = []

    if paths:
        path_right = paths[0]
        p_right = np.array([[int(x / scale_factor), int(y / scale_factor)] for y, x in path_right])
        left_top, left_bottom = [0, 0], [0, H-1]
        right_top, right_bottom = [p_right[0][0], 0], [p_right[-1][0], H-1]
        polygon = np.array([left_top, right_top] + p_right[::-1].tolist() + [right_bottom, left_bottom], dtype=np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        chunk = cv2.bitwise_and(image, image, mask=mask)
        x, y, w_box, h_box = cv2.boundingRect(polygon)
        cropped = chunk[y:y+h_box, x:x+w_box]
        fname = "chunk_00.png"
        cv2.imwrite(os.path.join(chunk_folder, fname), cropped)
        chunk_metadata.append({'filename': fname, 'x': int(x), 'y': int(y), 'width': int(w_box), 'height': int(h_box)})

        for i in range(len(paths) - 1):
            path1, path2 = paths[i], paths[i + 1]
            p1 = np.array([[int(x / scale_factor), int(y / scale_factor)] for y, x in path1])
            p2 = np.array([[int(x / scale_factor), int(y / scale_factor)] for y, x in path2])
            top1, top2 = [p1[0][0], 0], [p2[0][0], 0]
            bottom1, bottom2 = [p1[-1][0], H-1], [p2[-1][0], H-1]
            polygon = np.array([top1] + p1.tolist() + [bottom1] + [bottom2] + p2[::-1].tolist() + [top2], dtype=np.int32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            chunk = cv2.bitwise_and(image, image, mask=mask)
            x, y, w_box, h_box = cv2.boundingRect(polygon)
            cropped = chunk[y:y+h_box, x:x+w_box]
            fname = f"chunk_{i+1:02d}.png"
            cv2.imwrite(os.path.join(chunk_folder, fname), cropped)
            chunk_metadata.append({'filename': fname, 'x': int(x), 'y': int(y), 'width': int(w_box), 'height': int(h_box)})

        path_left = paths[-1]
        p_left = np.array([[int(x / scale_factor), int(y / scale_factor)] for y, x in path_left])
        left_top, left_bottom = [p_left[0][0], 0], [p_left[-1][0], H-1]
        right_top, right_bottom = [W-1, 0], [W-1, H-1]
        polygon = np.array([left_top] + p_left.tolist() + [left_bottom, right_bottom, right_top], dtype=np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        chunk = cv2.bitwise_and(image, image, mask=mask)
        x, y, w_box, h_box = cv2.boundingRect(polygon)
        cropped = chunk[y:y+h_box, x:x+w_box]
        fname = f"chunk_{len(paths):02d}.png"
        cv2.imwrite(os.path.join(chunk_folder, fname), cropped)
        chunk_metadata.append({'filename': fname, 'x': int(x), 'y': int(y), 'width': int(w_box), 'height': int(h_box)})

    else:
        fname = "chunk_00.png"
        cv2.imwrite(os.path.join(chunk_folder, fname), image)
        chunk_metadata.append({'filename': fname, 'x': 0, 'y': 0, 'width': W, 'height': H})

    # Use validated chunk_folder for meta_path
    meta_path = os.path.join(validated_chunk_folder, "chunk_coords.json")
    # Validate meta_path is within chunk_folder
    validated_meta_path = validate_path_within_directory(meta_path, validated_chunk_folder)
    if not validated_meta_path:
        raise ValueError(f"Invalid meta_path: {meta_path}")
    
    with open(validated_meta_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)

    log_duration("chunk_dewarped_image", timer, details=f"chunks={len(chunk_metadata)}")
    return validated_meta_path

# --- OCR & Annotation ---

def is_thick_image(chunk, threshold=120):
    """
    Return True for thick/ink-heavy, False for light/faint images.
    Adjust threshold as needed for your data.
    """
    if len(chunk.shape) == 3:
        gray = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
    else:
        gray = chunk
    mean_val = np.mean(gray)
    return mean_val < threshold

def enhance_image(input_path, output_path):
    chunk = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if chunk is None:
        return False
    _, binary = cv2.threshold(chunk, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    white_ratio = np.mean(binary == 255)
    if white_ratio < 0.92:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.dilate(binary, kernel, iterations=6)
    else:
        processed = binary
    cv2.imwrite(output_path, processed)
    return True

def enhance_image_alternative(input_path, output_path):
    chunk = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if chunk is None:
        return False
    _, binary = cv2.threshold(chunk, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    white_ratio = np.mean(binary == 255)
    if white_ratio < 0.92:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.dilate(binary, kernel, iterations=6)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.dilate(binary, kernel, iterations=5)
    h, w = processed.shape
    max_dim = 2500
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        processed = cv2.resize(processed, (new_w, new_h))
    cv2.imwrite(output_path, processed)
    return True

def enhance_image_toggle(input_path, output_path):
    chunk = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if chunk is None:
        return False
    if is_thick_image(chunk):
        return enhance_image(input_path, output_path)
    else:
        return enhance_image_alternative(input_path, output_path)

def extract_words_from_file(filepath):
    timer = start_timer()
    try:
        # Validate filepath to prevent path traversal
        if not filepath:
            app.logger.error("extract_words_from_file: Empty filepath provided")
            return []
        
        # Get the directory containing the file
        file_dir = os.path.dirname(os.path.abspath(filepath))
        
        # Validate path is within allowed directories (sessions or subdirectories)
        if not file_dir.startswith(os.path.abspath(SESSIONS_DIR)):
            app.logger.error(f"extract_words_from_file: Path traversal attempt detected: {filepath}")
            return []
        
        # Normalize and validate the path
        normalized_path = os.path.normpath(os.path.abspath(filepath))
        if not normalized_path.startswith(os.path.abspath(SESSIONS_DIR)):
            app.logger.error(f"extract_words_from_file: Invalid path after normalization: {filepath}")
            return []
        
        # Check if file exists
        if not os.path.exists(normalized_path) or not os.path.isfile(normalized_path):
            app.logger.error(f"extract_words_from_file: File does not exist: {normalized_path}")
            return []
        
        with open(normalized_path, "rb") as f:
            blob = f.read()
        if len(blob) > 10 * 1024 * 1024 or len(blob) == 0:
            app.logger.warning(f"File {os.path.basename(filepath)} is too large ({len(blob)} bytes) or empty")
            return []
        response = textract.detect_document_text(Document={"Bytes": blob})
        items = []
        for block in response.get("Blocks", []):
            if block["BlockType"] == "LINE":
                bbox = block["Geometry"]["BoundingBox"]
                items.append({
                    "text": block["Text"],
                    "bbox": {
                        "left": bbox["Left"],
                        "top": bbox["Top"],
                        "width": bbox["Width"],
                        "height": bbox["Height"],
                        "right": bbox["Left"] + bbox["Width"],
                        "bottom": bbox["Top"] + bbox["Height"]
                    }
                })
        log_duration("extract_words_from_file", timer, details=f"lines={len(items)} file={os.path.basename(filepath)}")
        return items
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Provide helpful error messages for common AWS credential issues
        if "UnrecognizedClientException" in error_type or "security token" in error_msg.lower():
            app.logger.error(f"extract_words_from_file failed for {os.path.basename(filepath)}: AWS CREDENTIALS EXPIRED/INVALID")
            app.logger.error(f"Error details: {error_type}: {error_msg}")
        else:
            app.logger.error(f"extract_words_from_file failed for {os.path.basename(filepath)}: {error_type}: {error_msg}")
        
        log_duration("extract_words_from_file", timer, details=f"failed file={os.path.basename(filepath)}, error={error_type}")
        return []

def map_chunk_coords_to_original(chunk_bbox, chunk_meta):
    info = chunk_meta
    x_offset = info['x']
    y_offset = info['y']
    chunk_w = info['width']
    chunk_h = info['height']
    left = chunk_bbox["left"] * chunk_w if chunk_bbox["left"] <= 1 else chunk_bbox["left"]
    top = chunk_bbox["top"] * chunk_h if chunk_bbox["top"] <= 1 else chunk_bbox["top"]
    width = chunk_bbox["width"] * chunk_w if chunk_bbox["width"] <= 1 else chunk_bbox["width"]
    height = chunk_bbox["height"] * chunk_h if chunk_bbox["height"] <= 1 else chunk_bbox["height"]
    mapped = {
        "left": left + x_offset,
        "top": top + y_offset,
        "width": width,
        "height": height,
        "right": left + x_offset + width,
        "bottom": top + y_offset + height
    }
    return mapped



def draw_bounding_boxes_on_chunk(original_image_path, enhanced_image_path, detected_items, output_path):
    """
    Draws bounding boxes and (always fully fitting or ellipsized) labels from detected_items
    on original_image_path, saving annotation to output_path.
    Each item in detected_items must have keys "bbox" and "text".
    """
    image = cv2.imread(original_image_path)
    if image is None:
        return False
    img_height, img_width = image.shape[:2]
    colors = [
        (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 165, 255),
        (128, 0, 128), (0, 255, 255), (255, 255, 0), (255, 0, 255)
    ]
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    MIN_FONT_SCALE = 0.35   # absolute minimum
    MAX_FONT_SCALE = 2.0

    for i, item in enumerate(detected_items):
        bbox = item["bbox"]
        text = item["text"]
        left = int(bbox["left"] * img_width)
        top = int(bbox["top"] * img_height)
        right = int(bbox["right"] * img_width)
        bottom = int(bbox["bottom"] * img_height)

        # Add some padding for visual clarity (optional, adjust as needed)
        pad_w = max(5, int((right - left) * 0.08))
        pad_h = max(5, int((bottom - top) * 0.18))
        left = max(0, left - pad_w)
        top = max(0, top - pad_h)
        right = min(img_width, right + pad_w)
        bottom = min(img_height, bottom + pad_h)
        box_width = max(right - left, 20)
        box_height = max(bottom - top, 16)

        # Draw bounding box
        color = colors[i % len(colors)]
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

        # ---- Fit text in box (shrink, then ellipsize if needed) ----
        display_text = text.strip()
        if not display_text:
            continue

        # Find largest font_scale that fits in width, but >= MIN_FONT_SCALE
        font_scale = MAX_FONT_SCALE
        thickness = max(1, box_height // 17)
        (text_width, text_height), _ = cv2.getTextSize(display_text, FONT_FACE, font_scale, thickness)
        while text_width > box_width - 10 and font_scale > MIN_FONT_SCALE:
            font_scale -= 0.04
            (text_width, text_height), _ = cv2.getTextSize(display_text, FONT_FACE, font_scale, thickness)

        # If at minimum font scale, still too wide: ellipsize string
        if text_width > box_width - 10:
            ellipsis = "..."
            max_chars = len(display_text)
            lo, hi = 1, max_chars
            while lo < hi:
                mid = (lo + hi) // 2
                test_text = display_text[:mid] + ellipsis
                (test_width, _), _ = cv2.getTextSize(test_text, FONT_FACE, MIN_FONT_SCALE, thickness)
                if test_width <= box_width - 10:
                    lo = mid + 1
                else:
                    hi = mid
            display_text = display_text[:hi-1] + ellipsis
            font_scale = MIN_FONT_SCALE
            (text_width, text_height), _ = cv2.getTextSize(display_text, FONT_FACE, font_scale, thickness)

        # Center text within the bounding box
        text_x = left + (box_width - text_width) // 2
        text_y = top + (box_height + text_height) // 2 - 2

        # Draw text background for contrast
        bg_top = text_y - text_height - 4
        bg_bot = text_y + 5
        bg_left = text_x - 6
        bg_right = text_x + text_width + 6
        cv2.rectangle(image, (bg_left, bg_top), (bg_right, bg_bot), (0,0,0), -1)
        cv2.putText(
            image, display_text, (text_x, text_y), FONT_FACE, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )
    cv2.imwrite(output_path, image)
    return True


def draw_on_original_image(original_image_path, all_detections, output_path):
    image = cv2.imread(original_image_path)
    if image is None:
        return False
    img_height, img_width = image.shape[:2]
    chunk_colors = {
        'chunk_00': (0, 0, 255), 'chunk_01': (255, 0, 0), 'chunk_02': (0, 255, 0), 'chunk_03': (0, 165, 255),
        'chunk_04': (128, 0, 128), 'chunk_05': (0, 255, 255), 'chunk_06': (255, 255, 0), 'chunk_07': (255, 0, 255),
        'chunk_08': (128, 128, 0), 'chunk_09': (0, 128, 128), 'chunk_10': (128, 0, 0),   'chunk_11': (0, 128, 0),
        'chunk_12': (0, 0, 128),   'chunk_13': (255, 192, 203), 'chunk_14': (165, 42, 42), 'chunk_15': (255, 165, 0),
    }
    for detection in all_detections:
        bbox = detection["original_bbox"]
        text = detection["text"]
        chunk_name = detection["chunk_name"]
        left = max(0, int(bbox["left"]))
        top = max(0, int(bbox["top"]))
        right = min(img_width, int(bbox["right"]))
        bottom = min(img_height, int(bbox["bottom"]))
        color = chunk_colors.get(chunk_name, (255, 255, 255))
        cv2.rectangle(image, (left, top), (right, bottom), color, 5)
        if text.strip():
            box_height = max(bottom - top, 40)
            font_scale = min(max(1.0, box_height / 18.0), 2.3)
            thickness = max(2, int(font_scale * 2.5))
            line_height = int(box_height * 0.75)
            max_text_width = right - left - 20
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = (current_line + " " + word).strip()
                (test_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                if test_width <= max_text_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            total_text_height = len(lines) * line_height
            start_y = top + max((box_height - total_text_height) // 2, 6)
            for j, line in enumerate(lines):
                (text_width, text_height), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = left + 10
                text_y = start_y + j * line_height
                pad_x = max(9, text_height // 2)
                pad_y = max(7, text_height // 3)
                cv2.rectangle(image, (text_x - pad_x, text_y - text_height - pad_y), (text_x + text_width + pad_x, text_y + pad_y), (0, 0, 0), -1)
                cv2.putText(image, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    cv2.imwrite(output_path, image)
    return True

import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from collections import OrderedDict

def normalize_value(val):
    if val is None:
        return ""
    return str(val).strip()

def is_unwanted_line(text):
    text = text.strip()
    if len(text) == 0:
        return True
    if len(text) <= 3 and not re.match(r'^E\s*\d{1,2}$', text, re.I):
        return True
    if re.fullmatch(r"[x\*\-\.=,~]+", text, re.I):
        return True
    if re.search(r"(certif|product quality|extra load|rotation|twi|week|date|quality)", text, re.I):
        return True
    return False

def format_plies_text(text):
    """
    Extract TREADWEAR separately, keep PLIES block formatted with line breaks
    """
    if not text:
        return []

    text = text.replace("&", "+")
    text = re.sub(r"\s+", " ", text).strip()

    entries = []

    # TREADWEAR
    treadwear_match = re.search(r"(TREADWEAR\s*\d+)", text, re.IGNORECASE)
    if treadwear_match:
        entries.append({"name": "UTQG TREADWEAR", "expected_text": treadwear_match.group(1)})
        text = text.replace(treadwear_match.group(1), "").strip()

    # PLIES
    if "PLIES" in text:
        text = re.sub(r"(PLIES\s*:)", r"\n\1", text, flags=re.IGNORECASE)
        text = re.sub(r"(SIDEWALL\s*:)", r"\n\1", text, flags=re.IGNORECASE)
        entries.append({"name": "PLIES", "expected_text": text.strip()})

    return entries

def extract_multiline_block_nlp(ocr_data, start_pattern, stop_patterns, min_sentence_length=30):
    start_idx = None
    for i, item in enumerate(ocr_data):
        if re.search(start_pattern, item['value'], re.I):
            start_idx = i
            break
    if start_idx is None:
        return [], -1

    lines = []
    idx = start_idx
    header_parts = []
    while idx < len(ocr_data):
        line = ocr_data[idx]['value'].strip()
        if not line:
            idx += 1
            continue
        if re.search(r'safety|warning', line, re.I):
            header_parts.append(line)
            idx += 1
            continue
        else:
            break
    header_text = " ".join(header_parts).replace("  ", " ").strip()
    if not header_text:
        header_text = "SAFETY WARNING"
    lines.append(header_text)

    while idx < len(ocr_data):
        line = ocr_data[idx]['value'].strip()
        if not line:
            break
        if any(re.search(pat, line, re.I) for pat in stop_patterns):
            break
        if is_unwanted_line(line):
            idx += 1
            continue
        if len(line) >= min_sentence_length or re.search(
                r'serious|injury|improper|tyre|failure|mounting|use|overload|trained|explosion', line, re.I):
            sentences = [s.strip() for s in sent_tokenize(line)]
            lines.extend([s for s in sentences if s])
        else:
            lines.append(line)
        idx += 1

    return lines, idx

def process_safety_warning_nlp(ocr_data, starting_id=1):
    stop_patterns = [
        r'BRAND\s*NAME', r'NOISE\s*NUMBER', r'DOT\s', r'LICENSE',
        r'ISI', r'ECE', r'MAX\s*LOAD', r'MAX\s*PRESS',
        r'^\d{2,3}/\d{2}', r'^[A-Z]?\d{3,}', r'INNER', r'READY'
    ]

    lines, end_idx = extract_multiline_block_nlp(ocr_data, r"(SAFETY|WARNING)", stop_patterns)
    if not lines:
        return [], starting_id

    entries = []
    proplist1 = [
        {"Name": "name", "Value": "SAFETY WARNING"},
        {"Name": "expected_text", "Value": normalize_value(lines[0])},
        {"Name": "control_id", "Value": "0"},
        {"Name": "side_a", "Value": "true"},
        {"Name": "side_b", "Value": "false"},
        {"Name": "instances", "Value": "1"}
    ]
    entries.append({"Properties": proplist1})
    if len(lines) > 1:
        val = "\n".join(lines[1:])
        proplist2 = [
            {"Name": "name", "Value": "SAFETY WARNING LINES"},
            {"Name": "expected_text", "Value": normalize_value(val)},
            {"Name": "control_id", "Value": "0"},
            {"Name": "side_a", "Value": "true"},
            {"Name": "side_b", "Value": "false"},
            {"Name": "instances", "Value": "1"}
        ]
        entries.append({"Properties": proplist2})
        return entries, starting_id + 2
    else:
        return entries, starting_id + 1

def createTireTemplate_clean(ocrData):
    entries = []
    id_counter = 1

    # SAFETY WARNING
    safety_entries, id_counter = process_safety_warning_nlp(ocrData, id_counter)
    entries.extend(safety_entries)
    used_lines = set()
    for e in entries:
        for prop in e["Properties"]:
            if prop["Name"] == "expected_text":
                used_lines.add(prop["Value"])

    # MADE IN / COUNTRY logic
    made_in, country = None, None
    for item in ocrData:
        val = item['value'].strip()
        if re.search(r"MADE\s+IN", val, re.I):
            made_in = val
        if re.search(r"\b(INDIA|USA|CHINA)\b", val, re.I):
            country = val
    made_in_value = None
    if made_in and country:
        made_in_value = f"{made_in}\n{country}"
    elif made_in:
        made_in_value = made_in
    elif country:
        made_in_value = country

    # Property patterns
    property_patterns = [
        {"name": "UTQG TRACTION", "regex": re.compile(r'TRACTION\s*[A-Z]', re.I)},
        {"name": "UTQG TEMPERATURE", "regex": re.compile(r'TEMPERATURE\s*[A-Z]', re.I)},
        {"name": "UTQG TREADWEAR/PLIES", "regex": re.compile(r'TREADWEAR.*PLIES', re.I)},
        {"name": "MADE IN", "regex": re.compile(r'MADE\s+IN|\b(INDIA|USA|CHINA)\b', re.I)},
        {"name": "NOISE NUMBER", "regex": re.compile(r'^\d{7}\s*[A-Z0-9]{2,}$|^\d{7}$')},
        {"name": "E MARK", "regex": re.compile(r'\bE\d+\b', re.I)},
        {"name": "TYRE SIZE", "regex": re.compile(r'\b\d{3}/\d{2}[-\s]*R[-\s]*\d{2}', re.I)},
        {"name": "BRAND NAME", "regex": re.compile(r'apollo|mrf|bridgestone|goodyear|vredstein|rionev|apterra|woods', re.I)},
        {"name": "DOT MARKING", "regex": re.compile(r'DOT\s+[A-Z0-9]+\s+[A-Z0-9]+', re.I)},
        {"name": "A CODE", "regex": re.compile(r'\bA\d+[-#A-Z0-9]*\b', re.I)},
        {"name": "ECE MARK", "regex": re.compile(r'\b\d{7,8}\b', re.I)},
        {"name": "MAX LOAD", "regex": re.compile(r'MAX\.?\s*LOAD.*', re.I)},
        {"name": "INFLATION PRESSURE", "regex": re.compile(r'MAX\.?\s*PRESS.*', re.I)},
        {"name": "TYRE TYPE", "regex": re.compile(r'RADIAL\s*TUBELESS', re.I)},
        {"name": "PLIES", "regex": re.compile(r'PLIES', re.I)},
    ]

    matched_lines = set()
    for pattern in property_patterns:
        for item in ocrData:
            val = item['value'].strip()
            if pattern["name"] == "MADE IN" and made_in_value:
                continue
            if val in used_lines or val in matched_lines or is_unwanted_line(val):
                continue

            if pattern["regex"].search(val):
                # Special handling for combined TREADWEAR + PLIES
                if pattern["name"] == "UTQG TREADWEAR/PLIES":
                    split_entries = format_plies_text(val)
                    for se in split_entries:
                        prop_list = [
                            {"Name": "name", "Value": se["name"]},
                            {"Name": "expected_text", "Value": normalize_value(se["expected_text"])},
                            {"Name": "control_id", "Value": "0"},
                            {"Name": "side_a", "Value": "true"},
                            {"Name": "side_b", "Value": "false"},
                            {"Name": "instances", "Value": "1"}
                        ]
                        entries.append({"Properties": prop_list})
                        id_counter += 1
                else:
                    prop_list = [
                        {"Name": "name", "Value": pattern["name"]},
                        {"Name": "expected_text", "Value": normalize_value(val)},
                        {"Name": "control_id", "Value": "0"},
                        {"Name": "side_a", "Value": "true"},
                        {"Name": "side_b", "Value": "false"},
                        {"Name": "instances", "Value": "1"}
                    ]
                    entries.append({"Properties": prop_list})
                    id_counter += 1

                matched_lines.add(val)

    # Insert single MADE IN combined
    if made_in_value:
        prop_list = [
            {"Name": "name", "Value": "MADE IN"},
            {"Name": "expected_text", "Value": normalize_value(made_in_value)},
            {"Name": "control_id", "Value": "0"},
            {"Name": "side_a", "Value": "true"},
            {"Name": "side_b", "Value": "false"},
            {"Name": "instances", "Value": "1"}
        ]
        entries.insert(len(safety_entries), {"Properties": prop_list})
        id_counter += 1

    # Add mandatory props
    user_id_counter = 1
    prop_order = [
        "camera", "control_id", "engraving", "expected_text", "instances",
        "mobile_stamp", "name", "side_a", "side_b", "upside_down", "user_id"
    ]

    for entry in entries:
        prop_dict = {p["Name"]: p["Value"] for p in entry["Properties"]}
        prop_dict.setdefault("camera", "SideWall")
        prop_dict.setdefault("control_id", "0")
        prop_dict.setdefault("engraving", "false")
        prop_dict.setdefault("mobile_stamp", "false")
        prop_dict.setdefault("side_a", "true")
        prop_dict.setdefault("side_b", "false")
        prop_dict.setdefault("upside_down", "false")
        prop_dict["user_id"] = str(user_id_counter)
        user_id_counter += 1
        entry["Properties"] = [
            {"Name": name, "Value": prop_dict.get(name, "")}
            for name in prop_order
        ]

    # Template name = brand + size
    brand, size = None, None
    for item in ocrData:
        val = item['value'].strip()
        if not brand and re.search(r'apollo|mrf|bridgestone|goodyear|vredstein|rionev|apterra|woods', val, re.I):
            brand = val
        if not size and re.search(r'\b\d{3}/\d{2}[-\s]*R[-\s]*\d{2}', val, re.I):
            size = val
    template_name = f"{(brand or '')} {(size or '')}".strip()

    final_json = OrderedDict()
    final_json["Name"] = template_name
    final_json["Entries"] = entries

    return final_json




# To serialize and print:
# result = createTireTemplate_clean(ocrData)
# print(json.dumps(result, indent=2))







# --- JSON Combining Utilities ---


def get_property_value(properties, key_name):
    key_name = key_name.lower()
    for prop in properties:
        if prop.get("Name", "").strip().lower() == key_name:
            return prop.get("Value")
    return None


def normalize_text_key(value):
    if value is None:
        return ""
    normalized = re.sub(r"\s+", " ", str(value).strip())
    return normalized.lower()


def merge_text_blocks(primary, secondary):
    primary_text = (primary or "").strip()
    secondary_text = (secondary or "").strip()

    if not primary_text:
        return secondary_text
    if not secondary_text:
        return primary_text

    if primary_text.lower() == secondary_text.lower():
        return primary_text
    if primary_text.lower() in secondary_text.lower():
        return secondary_text
    if secondary_text.lower() in primary_text.lower():
        return primary_text

    joiner = "\n" if ("\n" in primary_text or "\n" in secondary_text) else " | "
    return f"{primary_text}{joiner}{secondary_text}"


def merge_text_optional(base, addition):
    if not base:
        return addition
    if not addition:
        return base
    return merge_text_blocks(base, addition)


def combine_made_in_text(*values):
    parts = []
    for value in values:
        if not value:
            continue
        for fragment in re.split(r"\s*\|\s*|\n", value):
            cleaned = fragment.strip()
            if cleaned:
                parts.append(cleaned)
    ordered = []
    for part in parts:
        if part not in ordered:
            ordered.append(part)
    if not ordered:
        return ""
    return "\n".join(ordered)


def split_expected_text_units(value):
    if not value:
        return []
    return [
        token.strip()
        for token in re.split(r"(?:\s*\|\s*|\n)+", str(value))
        if token and token.strip()
    ]
def processsafetywarningnlpocrdata(nlpocrdata, startingid=1):
    stoppatterns = [
        r"BRAND NAME", r"NOISE NUMBER", r"DOT", r"LICENSE", r"TITLE", r"PLIES",
        r"ISI", r"ECE", r"MAX LOAD", r"MAX PRESS", r"r2,32", r"A-Z?3", r"INNER", r"READY"
    ]
    
    lines, endidx = extract_multiline_block_nlp(nlpocrdata, r"SAFETY WARNING", stoppatterns)
    if not lines:
        return [], startingid
    
    entries = []
    
    proplist1 = [
        {"Name": "name", "Value": "SAFETY WARNING"},
        {"Name": "expected_text", "Value": normalize_value(lines[0])},
        {"Name": "control_id", "Value": 0},
        {"Name": "side_a", "Value": True},
        {"Name": "side_b", "Value": False},
        {"Name": "camera", "Value": "SideWall"},
        {"Name": "engraving", "Value": False},
        {"Name": "instances", "Value": 1},
        {"Name": "mobile_stamp", "Value": False},
        {"Name": "upside_down", "Value": False},
        {"Name": "user_id", "Value": 2}
    ]
    entries.append({"Properties": proplist1})
    
    if len(lines) > 1:
        val = " ".join(lines[1:]).replace("|", "").strip()
        proplist2 = [
            {"Name": "name", "Value": "SAFETY WARNING LINES"},
            {"Name": "expected_text", "Value": normalize_value(val)},
            {"Name": "control_id", "Value": 0},
            {"Name": "side_a", "Value": True},
            {"Name": "side_b", "Value": False},
            {"Name": "camera", "Value": "SideWall"},
            {"Name": "engraving", "Value": False},
            {"Name": "instances", "Value": 1},
            {"Name": "mobile_stamp", "Value": False},
            {"Name": "upside_down", "Value": False},
            {"Name": "user_id", "Value": 2}
        ]
        entries.append({"Properties": proplist2})
        return entries, startingid + 2
    
    return entries, startingid + 1


def split_safety_warning_text(raw_text):
    tokens = split_expected_text_units(raw_text)
    header_tokens = [
        token for token in tokens
        if re.search(r"(safety|warning)", token, re.I)
        and not re.search(r"(serious|injury|tyre|persons|improper|overload|underinflation|allowed)", token, re.I)
    ]
    if not header_tokens:
        header_tokens = ["SAFETY WARNING"]
    header_text = re.sub(r"\s+", " ", " ".join(header_tokens)).strip()
    header_text = re.sub(r"[:\s]+$", "", header_text)
    if not header_text:
        header_text = "SAFETY WARNING"

    line_tokens = []
    skip_initial = True
    for token in tokens:
        if skip_initial:
            if re.search(r"^\s*(safety|warning)\b", token, re.I):
                continue
            skip_initial = False
        line_tokens.append(token)
    lines_text = "\n".join(line_tokens).strip()
    return header_text, lines_text


def normalize_expected_text_for_entry(entry_key, expected_text):
    if expected_text is None:
        return ""

    normalized_key = (entry_key or "").lower()
    text = str(expected_text).strip()
    if not text:
        return ""

    if normalized_key == "safety warning":
        header_text, _ = split_safety_warning_text(text)
        return header_text

    if normalized_key in {"safety warning lines", "safety warning line"}:
        _, lines_text = split_safety_warning_text(text)
        return lines_text

    if "plies" in normalized_key:
        return text.replace(" | ", "\n").strip()

    return text


def get_normalized_property(prop_dict, *key_names):
    for key_name in key_names:
        if not key_name:
            continue
        normalized = normalize_text_key(key_name)
        candidates = {
            normalized,
            normalized.replace(" ", "_"),
            normalized.replace("_", ""),
            normalized.replace(" ", ""),
        }
        for candidate in candidates:
            if candidate in prop_dict:
                return prop_dict[candidate]
    return None


EXPECTED_TEXT_NAME_OVERRIDES = {
    "205/65 r 16": "TYRE SIZE",
    "205/55 r16": "TYRE SIZE",
    "205/55 r 16": "TYRE SIZE",
    "205/55r16": "TYRE SIZE",
    "dot 042 3pc305": "DOT MARKING",
    "0211093 s2wr2": "NOISE NUMBER",
    "02116458": "ECE MARKING",
    "0414775 s2w2r3b": "ECE MARKING",
    "cm/l-6600053208": "ISI BOTTOM",
    "is:15633": "ISI TOP",
    "made in india": "MADE IN INDIA",
    "95": "LOAD INDEX",
    "95h": "95H",
    "h": "SPEED SYMBOL",
    "max. press: 350 kpa (51 psi)": "MAX INFL PRESSURE",
    "max press 350 kpa (51 psi)": "MAX INFL PRESSURE",
    "max. load: 690 kg (1521 lbs)": "MAX LOAD SINGLE",
    "max load 690 kg (1521 lbs)": "MAX LOAD SINGLE",
    "m+s": "M+S MARKING",
    "temperature a": "UTQG TEMPERATURE",
    "traction a": "UTQG TRACTION",
    "treadwear 460": "UTQG TREADWEAR",
    "apollo": "BRAND NAME",
    "apterra cross": "PRODUCT NAME",
    "ptterra cross": "PRODUCT NAME",
    "epollo": "BRAND NAME",
    "plies:": "PLIES",
    "tread: 1 polyester + 2 steel + 1 nylon": "PLIES AT TREAD",
    "tread: 2 polyester + 2 steel + 1 nylon": "PLIES AT TREAD",
    "tread: 1polyester + 2steel + 1nylon": "PLIES AT TREAD",
    "tread: 2polyester + 2 steel + 1nylon": "PLIES AT TREAD",
    "sidewall: 2 polyester": "PLIES SIDEWALL",
    "sidewall:2polyester": "PLIES SIDEWALL",
    "radial tubeless": "TYRE TYPE",
    "tread: 1 polyester + 2 steel + nylon": "PLIES AT TREAD",
    "sidewall:2 polyester": "PLIES SIDEWALL",
    "tread: 1 polyester + 2 steel + nylon": "PLIES AT TREAD",
    "****": "WEEK CODE",
    "a7541-##": "A CODE",
    "è": "E SYMBOL",
}


PATTERN_NAME_OVERRIDES = [
    (re.compile(r"safety\s*warning", re.I), "SAFETY WARNING"),
    (re.compile(r"\bwarning\b", re.I), "SAFETY WARNING"),
    (re.compile(r"serious\s+injury|improper\s+mounting|underinflation|overload|trained\s+persons|allowed\s+to\s+mount", re.I), "SAFETY WARNING LINES"),
    (re.compile(r"\bsafety\b", re.I), "SAFETY WARNING"),
    (re.compile(r"\bnoise\b|\bs2wr2\b", re.I), "NOISE NUMBER"),
    (re.compile(r"\bplies\b", re.I), "PLIES"),
    (re.compile(r"\bradial\s+tubeless\b", re.I), "TYRE TYPE"),
    (re.compile(r"\btreadwear\b", re.I), "UTQG TREADWEAR"),
    (re.compile(r"\btraction\b", re.I), "UTQG TRACTION"),
    (re.compile(r"\btemperature\b", re.I), "UTQG TEMPERATURE"),
    (re.compile(r"\bdot\b", re.I), "DOT MARKING"),
    (re.compile(r"\btread:\s*\d", re.I), "PLIES AT TREAD"),
    (re.compile(r"\bsidewall:\s*\d", re.I), "PLIES SIDEWALL"),
    (re.compile(r"\bmade\s+in\b", re.I), "MADE IN"),
    (re.compile(r"\bindia\b", re.I), "INDIA"),
]


def infer_entry_name(entry_name_value, expected_text):
    candidates = []
    if entry_name_value:
        candidates.append(entry_name_value)
    if expected_text:
        candidates.append(expected_text)

    for candidate in candidates:
        normalized = normalize_text_key(candidate)
        if normalized in EXPECTED_TEXT_NAME_OVERRIDES:
            return EXPECTED_TEXT_NAME_OVERRIDES[normalized]

    if expected_text:
        for pattern, inferred_name in PATTERN_NAME_OVERRIDES:
            if pattern.search(expected_text):
                return inferred_name

    return entry_name_value or expected_text


@app.route('/api/combine-tire-json', methods=['POST'])
def combine_tire_json():
    try:
        timer = start_timer()
        data = request.get_json(force=True)
        entries = data.get("Entries")
        custom_name = data.get("customName")

        if not entries or not isinstance(entries, list):
            log_duration("combine_tire_json", timer, details="invalid entries payload")
            return Response(json.dumps({"error": "Entries must be a non-empty list"}, indent=2), mimetype="application/json"), 400

        entries_by_name = OrderedDict()
        entry_counter = 0

        for entry in entries:
            props = entry.get("Properties", [])
            prop_dict = {}
            for p in props:
                name = p.get("Name")
                if not name:
                    continue
                prop_dict[normalize_text_key(name)] = p.get("Value")

            entry_name_value = get_normalized_property(prop_dict, "property_name", "name", "label", "title")

            if entry_name_value:
                entry_name_value = str(entry_name_value).strip()

            if not entry_name_value:
                entry_name_value = (get_normalized_property(prop_dict, "expected_text") or "").strip()

            if not entry_name_value:
                continue

            raw_expected_text = (get_normalized_property(prop_dict, "expected_text") or "").strip()

            display_name = infer_entry_name(entry_name_value, raw_expected_text)
            entry_key = normalize_text_key(display_name or entry_name_value or raw_expected_text)

            expected_text = normalize_expected_text_for_entry(entry_key, raw_expected_text)
            has_expected_text = bool(expected_text)
            safety_lines_text = ""
            if entry_key == "safety warning":
                _, safety_lines_text = split_safety_warning_text(raw_expected_text)

            side_a_val = get_normalized_property(prop_dict, "side_a", "sidea")
            side_b_val = get_normalized_property(prop_dict, "side_b", "sideb")
            side_a = str(side_a_val).strip().lower() == "true"
            side_b = str(side_b_val).strip().lower() == "true"

            if side_a_val is None and side_b_val is None:
                side_a = True
                side_b = True

            meta_defaults = {
                "camera": get_normalized_property(prop_dict, "camera") or "SideWall",
                "control_id": get_normalized_property(prop_dict, "control_id") or "0",
                "engraving": get_normalized_property(prop_dict, "engraving") or "false",
                "instances": get_normalized_property(prop_dict, "instances") or "1",
                "mobile_stamp": get_normalized_property(prop_dict, "mobile_stamp") or "false",
                "upside_down": get_normalized_property(prop_dict, "upside_down") or "false",
            }

            if entry_key not in entries_by_name:
                entries_by_name[entry_key] = {
                    "display_name": display_name,
                    "top_value": None,
                    "bottom_value": None,
                    "meta": dict(meta_defaults),
                    "index": entry_counter,
                }
                entry_counter += 1

            entry_info = entries_by_name[entry_key]
            if not entry_info.get("display_name"):
                entry_info["display_name"] = display_name

            for key, value in meta_defaults.items():
                entry_info["meta"].setdefault(key, value)

            if not has_expected_text:
                continue

            merge_allowed = entry_key in {"safety warning", "safety warning lines", "safety warning line"}

            if side_a:
                current_top = entry_info.get("top_value")
                if current_top and merge_allowed:
                    entry_info["top_value"] = merge_text_blocks(current_top, expected_text)
                elif not current_top:
                    entry_info["top_value"] = expected_text

            if side_b:
                current_bottom = entry_info.get("bottom_value")
                if current_bottom and merge_allowed:
                    entry_info["bottom_value"] = merge_text_blocks(current_bottom, expected_text)
                elif not current_bottom:
                    entry_info["bottom_value"] = expected_text

            if safety_lines_text:
                lines_key = "safety warning lines"
                lines_entry = entries_by_name.setdefault(
                    lines_key,
                    {
                        "display_name": "SAFETY WARNING LINES",
                        "top_value": None,
                        "bottom_value": None,
                        "meta": dict(meta_defaults),
                        "index": entry_counter,
                        "keep_separate": True,
                    }
                )
                if lines_entry.get("index") == entry_counter:
                    entry_counter += 1
                for key, value in meta_defaults.items():
                    lines_entry["meta"].setdefault(key, value)

                if side_a:
                    current_top = lines_entry.get("top_value")
                    lines_entry["top_value"] = merge_text_blocks(current_top, safety_lines_text) if current_top else safety_lines_text
                if side_b:
                    current_bottom = lines_entry.get("bottom_value")
                    lines_entry["bottom_value"] = merge_text_blocks(current_bottom, safety_lines_text) if current_bottom else safety_lines_text

        safety_key = None
        lines_key = None
        for key in list(entries_by_name.keys()):
            if key == "safety warning":
                safety_key = key
            elif key in {"safety warning lines", "safety warning line"}:
                lines_key = key

        if lines_key:
            lines_info = entries_by_name.pop(lines_key)
            if lines_info.get("keep_separate"):
                entries_by_name[lines_key] = lines_info
            elif safety_key is None:
                safety_key = "safety warning"
                lines_info["display_name"] = "SAFETY WARNING"
                entries_by_name[safety_key] = lines_info
            else:
                safety_info = entries_by_name[safety_key]
                safety_info["display_name"] = safety_info.get("display_name") or "SAFETY WARNING"
                safety_info["top_value"] = merge_text_blocks(safety_info.get("top_value"), lines_info.get("top_value"))
                safety_info["bottom_value"] = merge_text_blocks(safety_info.get("bottom_value"), lines_info.get("bottom_value"))
                if "index" in lines_info:
                    safety_info["index"] = min(safety_info.get("index", lines_info["index"]), lines_info["index"])
                for key, value in lines_info.get("meta", {}).items():
                    safety_info["meta"].setdefault(key, value)

        made_in_key = normalize_text_key("MADE IN")
        india_key = normalize_text_key("INDIA")
        combined_made_in_key = normalize_text_key("MADE IN INDIA")

        made_info = entries_by_name.pop(made_in_key, None)
        india_info = entries_by_name.pop(india_key, None)

        if made_info or india_info:
            existing_combined = entries_by_name.get(combined_made_in_key)
            base_meta = {}
            base_index = entry_counter

            for info in filter(None, [existing_combined, made_info, india_info]):
                for key, value in info.get("meta", {}).items():
                    base_meta.setdefault(key, value)
                base_index = min(base_index, info.get("index", base_index))

            combined_info = entries_by_name.setdefault(
                combined_made_in_key,
                {
                    "display_name": "MADE IN INDIA",
                    "top_value": None,
                    "bottom_value": None,
                    "meta": dict(base_meta),
                    "index": base_index,
                }
            )

            combined_info["display_name"] = "MADE IN INDIA"
            combined_info["index"] = min(combined_info.get("index", base_index), base_index)
            combined_info["meta"].update(base_meta)

            combined_info["top_value"] = combine_made_in_text(
                combined_info.get("top_value"),
                *(info.get("top_value") for info in [made_info, india_info] if info)
            )
            combined_info["bottom_value"] = combine_made_in_text(
                combined_info.get("bottom_value"),
                *(info.get("bottom_value") for info in [made_info, india_info] if info)
            )

            entries_by_name[combined_made_in_key] = combined_info

        if combined_made_in_key in entries_by_name:
            entries_by_name.pop(made_in_key, None)
            entries_by_name.pop(india_key, None)

        entries_info_list = sorted(entries_by_name.values(), key=lambda info: info.get("index", 0))

        merged_entries = []
        user_id_counter = 1

        for info in entries_info_list:
            top_value = (info.get("top_value") or "").strip()
            bottom_value = (info.get("bottom_value") or "").strip()

            if not top_value and not bottom_value:
                continue

            if top_value and bottom_value:
                final_value = merge_text_blocks(top_value, bottom_value)
                side_a_flag = True
                side_b_flag = True
            elif top_value:
                final_value = top_value
                side_a_flag = True
                side_b_flag = False
            else:
                final_value = bottom_value
                side_a_flag = False
                side_b_flag = True

            meta = info.get("meta", {})
            entry_display_name = info.get("display_name") or final_value
            entry_display_name = str(entry_display_name).strip()
            if not entry_display_name:
                entry_display_name = final_value

            if entry_display_name.upper() == "SAFETY WARNING":
                final_value = re.sub(r"[:\s]+$", "", final_value.strip())
                if not final_value:
                    final_value = "SAFETY WARNING"

            if entry_display_name.upper() == "SAFETY WARNING LINES":
                final_value = final_value.replace(" | ", "\n")

            if "plies" in entry_display_name.lower():
                final_value = final_value.replace(" | ", "\n")

            sanitized_final = re.sub(r"[\s\|\n]+", "", final_value)
            if len(sanitized_final) <= 1 and normalize_text_key(entry_display_name) == normalize_text_key(final_value):
                continue

            properties = [
                {"Name": "camera", "Value": str(meta.get("camera", "SideWall"))},
                {"Name": "control_id", "Value": str(meta.get("control_id", "0"))},
                {"Name": "engraving", "Value": str(meta.get("engraving", "false"))},
                {"Name": "expected_text", "Value": final_value},
                {"Name": "instances", "Value": str(meta.get("instances", "1"))},
                {"Name": "mobile_stamp", "Value": str(meta.get("mobile_stamp", "false"))},
                {"Name": "name", "Value": entry_display_name},
                {"Name": "side_a", "Value": "true" if side_a_flag else "false"},
                {"Name": "side_b", "Value": "true" if side_b_flag else "false"},
                {"Name": "upside_down", "Value": str(meta.get("upside_down", "false"))},
                {"Name": "user_id", "Value": str(user_id_counter)}
            ]

            merged_entries.append({"Properties": properties})
            user_id_counter += 1

        if not merged_entries:
            log_duration("combine_tire_json", timer, details="no merged entries")
            return Response(json.dumps({"error": "No valid entries to combine"}, indent=2), mimetype="application/json"), 400

        if custom_name and str(custom_name).strip():
            top_level_name = str(custom_name).strip()
        else:
            brand_name = None
            tyre_size = None
            for entry in merged_entries:
                props = entry.get("Properties", [])
                if not brand_name:
                    brand_name = get_property_value(props, "brand name")
                if not tyre_size:
                    tyre_size = get_property_value(props, "tyre size")
                if brand_name and tyre_size:
                    break

            if brand_name and tyre_size:
                top_level_name = f"{brand_name} {tyre_size}"
            elif brand_name:
                top_level_name = brand_name
            elif tyre_size:
                top_level_name = tyre_size
            else:
                top_level_name = "Combined Tire Specs"

        combined_json = OrderedDict([
            ("Name", top_level_name),
            ("Entries", merged_entries)
        ])

        response = Response(json.dumps(combined_json, indent=2), mimetype="application/json")
        log_duration("combine_tire_json", timer, details=f"merged_entries={len(merged_entries)}")
        return response

    except Exception as e:
        if 'timer' in locals():
            log_duration("combine_tire_json", timer, details="failed")
        error_json = OrderedDict([("error", str(e))])
        return Response(json.dumps(error_json, indent=2), mimetype="application/json"), 500













# --- Session & Rate Limiting Utilities ---


def get_ipstorage():
    if os.path.exists(IPSTORAGEFILE):
        with open(IPSTORAGEFILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}


def save_ipstorage(ipstorage):
    with open(IPSTORAGEFILE, "w") as f:
        json.dump(ipstorage, f)


def isipallowed(ip):
    now = time.time()
    ipstorage = get_ipstorage()
    ipinfo = ipstorage.get(ip, {'uploadcount': 0, 'lastrequesttime': 0})

    if now - ipinfo['lastrequesttime'] > COOLDOWN_PERIOD:
        ipinfo['uploadcount'] = 0
        ipinfo['lastrequesttime'] = now

    if ipinfo['uploadcount'] >= MAX_UPLOADS:
        ipstorage[ip] = ipinfo
        save_ipstorage(ipstorage)
        return False, COOLDOWN_PERIOD - (now - ipinfo['lastrequesttime'])

    ipinfo['uploadcount'] += 1
    ipinfo['lastrequesttime'] = now
    ipstorage[ip] = ipinfo
    save_ipstorage(ipstorage)
    return True, None


# --- Flask API Endpoints ---
LAST_SESSION_DIR = None


def resolve_session_path(session_id, tyre_type=None):
    """
    Resolve session path with security validation to prevent path traversal.
    """
    # Sanitize session_id
    sanitized_id = sanitize_session_id(session_id)
    if not sanitized_id:
        app.logger.error(f"Invalid session_id: {session_id}")
        return None
    
    # Build base path
    base_path = os.path.join(SESSIONS_DIR, sanitized_id)
    
    # Validate base path is within SESSIONS_DIR
    validated_base = validate_path_within_directory(base_path, SESSIONS_DIR)
    if not validated_base:
        app.logger.error(f"Path traversal attempt in session_id: {session_id}")
        return None
    
    # Handle tyre_type subdirectory
    if tyre_type and tyre_type.lower() in ["top", "bottom"]:
        sub_path = os.path.join(validated_base, tyre_type.lower())
        # Validate subdirectory path
        validated_sub = validate_path_within_directory(sub_path, validated_base)
        if not validated_sub:
            app.logger.error(f"Path traversal attempt in tyre_type: {tyre_type}")
            return None
        return validated_sub
    
    return validated_base



@app.route('/upload', methods=['POST'])
def upload():
    global LAST_SESSION_DIR

    ip = request.remote_addr
    allowed, wait_time = isipallowed(ip)
    if not allowed:
        return jsonify(error=f"Rate limit exceeded. Try again after {int(wait_time)} seconds."), 429

    if 'file' not in request.files:
        return jsonify(error="No file uploaded"), 400
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify(error="Empty filename"), 400

    image_type = request.form.get('image_type', 'light').lower()
    if image_type not in ['thick', 'light']:
        return jsonify(error="Invalid image_type. Must be 'thick' or 'light'"), 400

    tyre_type = request.form.get('tyre_type', 'single').lower()
    if tyre_type not in ['top', 'bottom', 'single']:
        return jsonify(error="Invalid tyre_type. Must be 'top', 'bottom', or 'single'"), 400

    provided_session_id = request.form.get('session_id')
    if provided_session_id:
        # Sanitize provided session_id
        sess_id = sanitize_session_id(provided_session_id.strip())
        if not sess_id:
            # If sanitization fails, generate new one
            sess_id = uuid.uuid4().hex
    else:
        sess_id = uuid.uuid4().hex

    # Resolve and validate session path
    session_path = resolve_session_path(sess_id, None if tyre_type == 'single' else tyre_type)
    if not session_path:
        return jsonify(error="Invalid session_id"), 400
    
    os.makedirs(session_path, exist_ok=True)

    # Secure filename handling
    filename = secure_filename(f.filename)
    if not filename:
        return jsonify(error="Invalid filename"), 400
    
    # Create secure file path
    input_path = secure_file_path(session_path, filename)
    if not input_path:
        return jsonify(error="Invalid file path"), 400
    f.save(input_path)

    overlay_path = os.path.join(session_path, "overlay.jpg")
    dewarp_path = os.path.join(session_path, "dewarp.jpg")
    chunk_dir = os.path.join(session_path, "chunks")
    enhanced_dir = os.path.join(session_path, "enhanced")
    ocr_output_dir = os.path.join(session_path, "ocr_output")
    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(ocr_output_dir, exist_ok=True)

    process_timer = start_timer()
    try:
        img_color = cv2.imread(input_path, cv2.IMREAD_COLOR)
        img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img_color is None or img_gray is None:
            return jsonify(error="Failed to read uploaded image"), 500

        step_timer = start_timer()
        detected = detect_circle_by_horizontal_scan(input_path)
        if not detected:
            return jsonify(error="No circle detected"), 400
        cx, cy, r = detected[0]
        log_duration("circle_detection", step_timer, details=f"radius={r}")

        overlay = img_color.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imwrite(overlay_path, overlay)

        adaptive_thresh = cv2.adaptiveThreshold(
            img_gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31, 15
        )

        dilated_mask = cv2.dilate(adaptive_thresh, np.ones((10, 10), np.uint8), iterations=10)
        cv2.imwrite(os.path.join(session_path, "binary_for_angle.png"), dilated_mask)

        step_timer = start_timer()
        dewarp_circle_to_rect(img_color, (cx, cy, r), dilated_mask, dewarp_path, stretch=3)
        log_duration("dewarp_circle_to_rect (upload wrapper)", step_timer)

        step_timer = start_timer()
        chunk_coords_json = chunk_dewarped_image(dewarp_path, chunk_dir)
        # Validate chunk_coords_json path before reading
        if not chunk_coords_json:
            return jsonify(error="Failed to generate chunk coordinates"), 500
        
        validated_chunk_json = validate_path_within_directory(chunk_coords_json, SESSIONS_DIR)
        if not validated_chunk_json:
            return jsonify(error="Invalid chunk coordinates path"), 500
        
        with open(validated_chunk_json, "r") as f:
            chunk_coords = json.load(f)
        log_duration("chunk_dewarped_image (upload wrapper)", step_timer, details=f"chunks={len(chunk_coords)}")

        all_detections = []

        for chunk_meta in chunk_coords:
            chunk_timer = start_timer()
            fname = chunk_meta.get("filename", "")
            if not fname:
                continue
            
            # Sanitize and validate chunk filenames
            safe_fname = secure_filename(fname)
            if not safe_fname:
                app.logger.warning(f"Invalid chunk filename: {fname}")
                continue
            
            # Create secure paths for chunk files
            raw_chunk_path = secure_file_path(chunk_dir, safe_fname)
            enhanced_chunk_path = secure_file_path(enhanced_dir, safe_fname)
            
            if not raw_chunk_path or not enhanced_chunk_path:
                app.logger.warning(f"Path validation failed for chunk: {fname}")
                continue

            if image_type == 'thick':
                print(f"Processing {fname} as THICK image - no enhancement applied")
                shutil.copyfile(raw_chunk_path, enhanced_chunk_path)
            else:
                print(f"Processing {fname} as LIGHT image - applying enhancement")
                if not enhance_image(raw_chunk_path, enhanced_chunk_path):
                    enhance_image_alternative(raw_chunk_path, enhanced_chunk_path)

            detected_items = extract_words_from_file(enhanced_chunk_path)
            if not detected_items and image_type == 'light':
                enhance_image_alternative(raw_chunk_path, enhanced_chunk_path)
                detected_items = extract_words_from_file(enhanced_chunk_path)

            # Secure annotated chunk path
            annotated_fname = safe_fname.replace(".png", "_annotated.png")
            annotated_chunk_path = secure_file_path(ocr_output_dir, annotated_fname)
            if not annotated_chunk_path:
                app.logger.warning(f"Path validation failed for annotated chunk: {annotated_fname}")
                continue
            
            draw_bounding_boxes_on_chunk(raw_chunk_path, enhanced_chunk_path, detected_items, annotated_chunk_path)

            for item in detected_items:
                original_bbox = map_chunk_coords_to_original(item["bbox"], chunk_meta)
                all_detections.append({
                    "text": item["text"],
                    "chunk_bbox": item["bbox"],
                    "original_bbox": original_bbox,
                    "chunk_name": os.path.splitext(safe_fname)[0]
                })
            log_duration("chunk_processing", chunk_timer, details=f"{fname}, detections={len(detected_items)}")

        # Secure paths for output files
        annotated_original_path = secure_file_path(session_path, "annotated_full_original.png")
        if not annotated_original_path:
            return jsonify(error="Invalid path for annotated image"), 500
        
        draw_on_original_image(dewarp_path, all_detections, annotated_original_path)

        ocr_json_path = secure_file_path(session_path, "ocr.json")
        if not ocr_json_path:
            return jsonify(error="Invalid path for OCR JSON"), 500
        
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            json.dump(all_detections, f, indent=2, ensure_ascii=False)

        LAST_SESSION_DIR = session_path

        if tyre_type in ['top', 'bottom']:
            annotated_endpoint = f"/session/{sess_id}/{tyre_type}/annotated"
            dewarped_endpoint = f"/session/{sess_id}/{tyre_type}/dewarped"
            ocr_endpoint = f"/session/{sess_id}/{tyre_type}/ocr"
        else:
            annotated_endpoint = f"/session/{sess_id}/annotated"
            dewarped_endpoint = f"/session/{sess_id}/dewarped"
            ocr_endpoint = f"/session/{sess_id}/ocr"

        log_duration("full_upload_pipeline", process_timer, details=f"session={sess_id}, detections={len(all_detections)}")

        return jsonify({
            "session_id": sess_id,
            "image_type": image_type,
            "tyre_type": tyre_type,
            "annotated_url": annotated_endpoint,
            "dewarped_url": dewarped_endpoint,
            "ocr_url": ocr_endpoint,
            "detections": all_detections
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        log_duration("full_upload_pipeline", process_timer, details="failed")
        return jsonify(error=f"Processing error: {str(e)}"), 500


@app.route('/dewarped')
def get_dewarped():
    if not LAST_SESSION_DIR:
        return jsonify(error="none yet"), 404
    
    # Validate LAST_SESSION_DIR is within SESSIONS_DIR
    validated_session_dir = validate_path_within_directory(LAST_SESSION_DIR, SESSIONS_DIR)
    if not validated_session_dir:
        return jsonify(error="Invalid session directory"), 400
    
    path = secure_file_path(validated_session_dir, "dewarp.jpg")
    if not path or not os.path.exists(path):
        return jsonify(error="dewarped image not found"), 404
    return send_file(path, mimetype="image/jpeg")

@app.route('/stitched')
def get_stitched():
    if not LAST_SESSION_DIR:
        return jsonify(error="none yet"), 404
    
    # Validate LAST_SESSION_DIR is within SESSIONS_DIR
    validated_session_dir = validate_path_within_directory(LAST_SESSION_DIR, SESSIONS_DIR)
    if not validated_session_dir:
        return jsonify(error="Invalid session directory"), 400
    
    path = secure_file_path(validated_session_dir, "annotated_full_original.png")
    if not path or not os.path.exists(path):
        return jsonify(error="stitched image not found"), 404
    return send_file(path, mimetype="image/png")


@app.route('/session/<session_id>/annotated')
def get_session_annotated(session_id):
    session_path = resolve_session_path(session_id)
    if not session_path:
        return jsonify(error="Invalid session_id"), 400
    
    path = secure_file_path(session_path, "annotated_full_original.png")
    if not path or not os.path.exists(path):
        return jsonify(error="annotated image not found"), 404
    return send_file(path, mimetype="image/png")


@app.route('/session/<session_id>/dewarped')
def get_session_dewarped(session_id):
    session_path = resolve_session_path(session_id)
    if not session_path:
        return jsonify(error="Invalid session_id"), 400
    
    path = secure_file_path(session_path, "dewarp.jpg")
    if not path or not os.path.exists(path):
        return jsonify(error="dewarped image not found"), 404
    return send_file(path, mimetype="image/jpeg")


@app.route('/session/<session_id>/ocr')
def get_session_ocr(session_id):
    session_path = resolve_session_path(session_id)
    if not session_path:
        return jsonify(error="Invalid session_id"), 400
    
    path = secure_file_path(session_path, "ocr.json")
    if not path or not os.path.exists(path):
        return jsonify(error="ocr data not found"), 404
    return send_file(path, mimetype="application/json")


@app.route('/session/<session_id>/<tyre_type>/annotated')
def get_session_annotated_tyre(session_id, tyre_type):
    if tyre_type not in ['top', 'bottom']:
        return jsonify(error="Invalid tyre_type"), 400
    
    session_path = resolve_session_path(session_id, tyre_type)
    if not session_path:
        return jsonify(error="Invalid session_id"), 400
    
    path = secure_file_path(session_path, "annotated_full_original.png")
    if not path or not os.path.exists(path):
        return jsonify(error="annotated image not found"), 404
    return send_file(path, mimetype="image/png")


@app.route('/session/<session_id>/<tyre_type>/dewarped')
def get_session_dewarped_tyre(session_id, tyre_type):
    if tyre_type not in ['top', 'bottom']:
        return jsonify(error="Invalid tyre_type"), 400
    
    session_path = resolve_session_path(session_id, tyre_type)
    if not session_path:
        return jsonify(error="Invalid session_id"), 400
    
    path = secure_file_path(session_path, "dewarp.jpg")
    if not path or not os.path.exists(path):
        return jsonify(error="dewarped image not found"), 404
    return send_file(path, mimetype="image/jpeg")


@app.route('/session/<session_id>/<tyre_type>/ocr')
def get_session_ocr_tyre(session_id, tyre_type):
    if tyre_type not in ['top', 'bottom']:
        return jsonify(error="Invalid tyre_type"), 400
    
    session_path = resolve_session_path(session_id, tyre_type)
    if not session_path:
        return jsonify(error="Invalid session_id"), 400
    
    path = secure_file_path(session_path, "ocr.json")
    if not path or not os.path.exists(path):
        return jsonify(error="ocr data not found"), 404
    return send_file(path, mimetype="application/json")

@app.route('/ocr')
def get_ocr():
    if not LAST_SESSION_DIR:
        return jsonify(error="none yet"), 404
    
    # Validate LAST_SESSION_DIR is within SESSIONS_DIR
    validated_session_dir = validate_path_within_directory(LAST_SESSION_DIR, SESSIONS_DIR)
    if not validated_session_dir:
        return jsonify(error="Invalid session directory"), 400
    
    path = secure_file_path(validated_session_dir, "ocr.json")
    if not path or not os.path.exists(path):
        return jsonify(error="ocr data not found"), 404
    return send_file(path, mimetype="application/json")

@app.route('/get_bbox', methods=["POST"])
def get_bbox():
    if not LAST_SESSION_DIR:
        return jsonify(error="No session"), 404
    
    # Validate LAST_SESSION_DIR is within SESSIONS_DIR
    validated_session_dir = validate_path_within_directory(LAST_SESSION_DIR, SESSIONS_DIR)
    if not validated_session_dir:
        return jsonify(error="Invalid session directory"), 400
    
    path = secure_file_path(validated_session_dir, "ocr.json")
    if not path or not os.path.exists(path):
        return jsonify(error="ocr data not found"), 404
    
    req = request.get_json()
    text = req.get("text")
    chunk_name = req.get("chunk_name")  # optionally, frontend could pass this
    # Do NOT trust index - indexes will diverge after filtering/merging!
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Prefer exact match by BOTH text and chunk_name if chunk_name exists
    match = None
    if text:
        # Try to match by both text and chunk_name if chunk_name is provided
        if chunk_name:
            match = next((item for item in data if item["text"].strip() == text.strip() and item.get("chunk_name") == chunk_name), None)
        if not match:
            # fallback: just match by text (first occurrence)
            match = next((item for item in data if item["text"].strip() == text.strip()), None)
    # 2. If still no match, fallback to index ONLY AS LAST RESORT
    if not match and "index" in req:
        try:
            match = data[req["index"]]
        except Exception:
            pass
    if not match:
        return jsonify(error="not found"), 404

    return jsonify({
        "original_bbox": match["original_bbox"],
        "chunk_bbox": match.get("chunk_bbox"),
        "chunk_name": match.get("chunk_name"),
        "text": match["text"]
    })
@app.route('/api/image-types', methods=['GET'])
def get_image_types():
    """Get available image types for the toggle"""
    return jsonify({
        "image_types": [
            {
                "value": "thick",
                "label": "Thick Image",
                "description": "No enhancement applied - direct OCR on raw chunks"
            },
            {
                "value": "light", 
                "label": "Light Image",
                "description": "Enhancement applied before OCR processing"
            }
        ]
    })

@app.route('/api/process-tire-ocr', methods=['POST'])
def process_tire_ocr():
    try:
        timer = start_timer()
        ocr_data = request.json
        if ocr_data is None:
            return jsonify({"error": "No JSON data received"}), 400
        if not isinstance(ocr_data, list):
            return jsonify({"error": "Expected a JSON array of OCR data"}), 400

        # Normalize input
        processed_ocr_data = []
        for item in ocr_data:
            if isinstance(item, dict) and 'value' in item:
                processed_ocr_data.append({'value': item['value']})
            elif isinstance(item, str):
                processed_ocr_data.append({'value': item})
            else:
                return jsonify({"error": "Invalid OCR data format"}), 400

        # Generate template
        template = createTireTemplate_clean(processed_ocr_data)

        # Preserve order in response
        response = Response(json.dumps(template, indent=2), mimetype="application/json")
        log_duration("process_tire_ocr", timer, details=f"entries={len(template)}")
        return response

    except Exception as e:
        traceback.print_exc()
        if 'timer' in locals():
            log_duration("process_tire_ocr", timer, details="failed")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    



# merge_entries function remains unchanged...



def merge_entries(entries):
    grouped = {}
    for entry in entries:
        props = entry.get('Properties', [])
        main_name = None
        for p in props:
            if p.get('Name', '').lower() == 'name':
                main_name = p.get('Value')
                break
        if not main_name:
            main_name = f"Unknown_{len(grouped)}"

        if main_name not in grouped:
            grouped[main_name] = OrderedDict()

        merged_props = grouped[main_name]

        for prop in props:
            pname = prop.get('Name')
            pvalue = prop.get('Value')
            if pname is None:
                continue

            if pname.lower() in ['side_a', 'sidea', 'side_b', 'sideb']:
                existing_val = merged_props.get(pname, "false").lower()
                new_val = str(pvalue).lower()
                merged_props[pname] = 'true' if existing_val == 'true' or new_val == 'true' else 'false'
            else:
                if pname not in merged_props:
                    merged_props[pname] = pvalue

    merged_entries = []
    for main_name, props_dict in grouped.items():
        properties_list = [{"Name": k, "Value": v} for k, v in props_dict.items()]
        merged_entries.append({"Properties": properties_list})

    return merged_entries







if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
