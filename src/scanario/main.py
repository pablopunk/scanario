#!/usr/bin/env python3
"""scanario – detect the 4 corners of a paper document in a phone photo.

Pipeline:
  1. Ask Gemini 2.5 Flash Image ("nano banana") to repaint everything that is
     not the main paper sheet with solid magenta, preserving the paper pixels.
  2. Threshold by HSV saturation → a rough binary mask of the paper.
  3. Find the 4 dominant straight lines in the mask boundary with successive
     RANSAC. Intersect them to recover even corners that are occluded by an
     object lying on top of the paper (e.g. a small receipt).
"""

import argparse
import io
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from rembg import remove, new_session

load_dotenv()
SCRIPT_DIR = Path(__file__).resolve().parent

NANO_BANANA_PROMPT = (
    "Edit this image with a STRICT pixel-preserving background replacement.\n\n"
    "There is one large white paper document on a table. A small receipt rests on top.\n\n"
    "Task: replace every pixel that is NOT part of the large paper document with "
    "solid pure magenta (#FF00FF). The small receipt counts as part of the document "
    "and must remain unchanged.\n\n"
    "Hard constraints:\n"
    "- Keep the document and receipt in EXACTLY the same position, scale, perspective, and shape.\n"
    "- Do NOT redraw, restyle, enhance, sharpen, or regenerate the document.\n"
    "- Do NOT move any edge or corner even by 1 pixel.\n"
    "- Do NOT crop, rotate, zoom, or change framing.\n"
    "- Keep the exact same image dimensions.\n"
    "- Only background pixels may change, and they must become solid #FF00FF."
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return 4 points in TL, TR, BR, BL order."""
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


# ---------------------------------------------------------------------------
# Step 1: geometry-preserving isolation backends
# ---------------------------------------------------------------------------

REMBG_SESSION = None


def isolate_with_nano_banana(img_bgr: np.ndarray) -> np.ndarray:
    """Send the image to Gemini 2.5 Flash Image and receive a magenta-background version."""
    client = genai.Client()
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[NANO_BANANA_PROMPT, pil],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    for cand in response.candidates:
        for part in cand.content.parts:
            if part.inline_data and part.inline_data.data:
                out = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    raise RuntimeError("Nano Banana returned no image")


def isolate_with_rembg(img_bgr: np.ndarray) -> np.ndarray:
    global REMBG_SESSION
    if REMBG_SESSION is None:
        REMBG_SESSION = new_session("u2net")
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    out = remove(pil, session=REMBG_SESSION).convert("RGBA")
    rgba = np.array(out)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    black = np.zeros_like(rgb)
    comp = (rgb.astype(np.float32) * alpha + black.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)


def estimate_geometry_drift(original_bgr: np.ndarray, isolated_bgr: np.ndarray, debug_dir: Path = None):
    """Estimate whether Nano Banana preserved geometry well enough.

    We match ORB features mostly on low-saturation (document/receipt) content.
    Returns dict with ok flag, reprojection error, and match counts.
    """
    if isolated_bgr.shape[:2] != original_bgr.shape[:2]:
        isolated_bgr = cv2.resize(isolated_bgr, (original_bgr.shape[1], original_bgr.shape[0]))

    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    iso_gray = cv2.cvtColor(isolated_bgr, cv2.COLOR_BGR2GRAY)
    iso_hsv = cv2.cvtColor(isolated_bgr, cv2.COLOR_BGR2HSV)
    # Focus on document-ish area: low saturation in the isolated image.
    mask = (iso_hsv[:, :, 1] < 80).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(orig_gray, mask)
    kp2, des2 = orb.detectAndCompute(iso_gray, mask)
    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return {"ok": False, "reason": "too_few_features", "inliers": 0, "rmse": 1e9}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 20:
        return {"ok": False, "reason": "too_few_matches", "inliers": len(good), "rmse": 1e9}

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inlier_mask is None:
        return {"ok": False, "reason": "homography_failed", "inliers": 0, "rmse": 1e9}

    inliers = inlier_mask.ravel().astype(bool)
    src_in = src[inliers]
    dst_in = dst[inliers]
    proj = cv2.perspectiveTransform(src_in, H)
    err = np.linalg.norm((proj - dst_in).reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else 1e9
    ok = len(src_in) >= 30 and rmse <= 2.5

    if debug_dir:
        match_vis = cv2.drawMatches(
            original_bgr, kp1, isolated_bgr, kp2,
            [good[i] for i in range(min(80, len(good)))], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(debug_dir / "debug_matches.jpg"), match_vis)
    return {"ok": ok, "reason": "ok" if ok else "geometry_drift", "inliers": int(len(src_in)), "rmse": rmse}


# ---------------------------------------------------------------------------
# Step 2: Saturation-based mask
# ---------------------------------------------------------------------------

def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask


def mask_from_isolated(isolated_bgr: np.ndarray, method: str) -> np.ndarray:
    if method == "nano":
        hsv = cv2.cvtColor(isolated_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mask = (sat < 60).astype(np.uint8)
        return postprocess_mask(mask)
    if method == "rembg":
        gray = cv2.cvtColor(isolated_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray > 8).astype(np.uint8)
        return postprocess_mask(mask)
    raise ValueError(f"Unknown mask method: {method}")


def quad_iou_score(mask: np.ndarray, quad: np.ndarray) -> float:
    qm = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(qm, np.round(quad).astype(np.int32), 1)
    inter = np.logical_and(mask > 0, qm > 0).sum()
    union = np.logical_or(mask > 0, qm > 0).sum()
    return float(inter / union) if union else 0.0


# ---------------------------------------------------------------------------
# Step 3: 4-line RANSAC quad fit
# ---------------------------------------------------------------------------

def _ransac_line(pts: np.ndarray, inlier_dist: float, n_iter: int, rng: np.random.Generator):
    """Return (direction, point_on_line, inlier_mask) for the best line."""
    if len(pts) < 10:
        return None
    N = len(pts)
    best = None
    for _ in range(n_iter):
        i, j = rng.choice(N, size=2, replace=False)
        p1 = pts[i]
        p2 = pts[j]
        d = p2 - p1
        L = float(np.linalg.norm(d))
        if L < 5:
            continue
        d = d / L
        n = np.array([-d[1], d[0]])
        resid = np.abs((pts - p1) @ n)
        inliers = resid < inlier_dist
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, d, p1, inliers)
    if best is None:
        return None

    _, d, p1, inliers = best
    if inliers.sum() >= 2:
        vx, vy, x0, y0 = cv2.fitLine(pts[inliers], cv2.DIST_L2, 0, 0.01, 0.01)
        d = np.array([float(vx[0]), float(vy[0])])
        p1 = np.array([float(x0[0]), float(y0[0])])
    return d, p1, inliers


def _intersect(l1, l2):
    d1, p1 = l1
    d2, p2 = l2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = p2 - p1
    try:
        t, _ = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return p1 + t * d1


def fit_quad(mask: np.ndarray):
    """Fit a quadrilateral to the mask by finding 4 dominant boundary lines."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

    # Drop contour points that lie on the image border: if the paper is cut off,
    # those segments are not real paper edges.
    H, W = mask.shape[:2]
    margin = 2
    in_frame = (
        (contour[:, 0] > margin)
        & (contour[:, 0] < W - 1 - margin)
        & (contour[:, 1] > margin)
        & (contour[:, 1] < H - 1 - margin)
    )
    pts = contour[in_frame] if in_frame.sum() > 40 else contour

    inlier_dist = max(3.0, 0.003 * max(H, W))
    rng = np.random.default_rng(42)
    remaining = pts.copy()
    lines = []
    for _ in range(4):
        if len(remaining) < 10:
            break
        res = _ransac_line(remaining, inlier_dist=inlier_dist, n_iter=800, rng=rng)
        if res is None:
            break
        d, p0, inliers = res
        lines.append((d, p0))
        remaining = remaining[~inliers]

    if len(lines) < 4:
        return None

    # Split the 4 lines into two orientation groups: near-horizontal vs near-vertical.
    horiz = [ln for ln in lines if abs(ln[0][0]) > abs(ln[0][1])]
    vert = [ln for ln in lines if abs(ln[0][0]) <= abs(ln[0][1])]
    if len(horiz) != 2 or len(vert) != 2:
        # Fallback: split by angle mod 180° at the largest gap.
        angles = np.array([np.degrees(np.arctan2(d[1], d[0])) % 180 for d, _ in lines])
        order = np.argsort(angles)
        sorted_angles = angles[order]
        gaps = np.diff(sorted_angles)
        wrap = (sorted_angles[0] + 180) - sorted_angles[-1]
        all_gaps = list(gaps) + [wrap]
        split = int(np.argmax(all_gaps))
        g1 = [lines[order[k]] for k in range(split + 1)]
        g2 = [lines[order[k]] for k in range(split + 1, 4)]
        if len(g1) != 2 or len(g2) != 2:
            return None
        # Decide which group is horizontal.
        if abs(g1[0][0][0]) > abs(g1[0][0][1]):
            horiz, vert = g1, g2
        else:
            horiz, vert = g2, g1

    # Within each group, the line with smaller y (or x) origin is "top" (or "left").
    horiz.sort(key=lambda ln: ln[1][1])
    vert.sort(key=lambda ln: ln[1][0])
    top_line, bot_line = horiz
    left_line, right_line = vert

    corners = [
        _intersect(top_line, left_line),
        _intersect(top_line, right_line),
        _intersect(bot_line, right_line),
        _intersect(bot_line, left_line),
    ]
    if any(c is None for c in corners):
        return None

    corners = np.array(corners, dtype=np.float32)

    # If line intersections land outside the image, fall back to a polygonal hull
    # built from in-frame contour points only. This avoids spurious vertices caused
    # by masks leaking into the image border.
    if np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= W) or \
       np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= H):
        hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2).astype(np.float32)
        peri = cv2.arcLength(hull.reshape(-1, 1, 2), True)
        approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 0.01 * peri, True).reshape(-1, 2).astype(np.float32)

        border_margin = max(8.0, 0.01 * max(H, W))
        keep = (
            (approx[:, 0] > border_margin)
            & (approx[:, 0] < W - 1 - border_margin)
            & (approx[:, 1] > border_margin)
            & (approx[:, 1] < H - 1 - border_margin)
        )
        candidates = approx[keep]
        if len(candidates) < 4:
            candidates = approx

        if len(candidates) == 4:
            corners = candidates
        elif len(candidates) > 4:
            import itertools
            best_quad = None
            best_score = -1e9
            for combo in itertools.combinations(range(len(candidates)), 4):
                quad = order_points(candidates[list(combo)])
                if cv2.contourArea(quad.astype(np.float32)) < 100:
                    continue
                score = quad_iou_score(mask, quad)
                # Slightly prefer quads whose corners are away from the border.
                d = np.minimum.reduce([
                    quad[:, 0],
                    quad[:, 1],
                    (W - 1) - quad[:, 0],
                    (H - 1) - quad[:, 1],
                ])
                score += 0.0005 * float(np.mean(d))
                if score > best_score:
                    best_score = score
                    best_quad = quad
            if best_quad is not None:
                corners = best_quad

    return order_points(corners)


# ---------------------------------------------------------------------------
# Step 4: refine the 4 sides on the ORIGINAL image
# ---------------------------------------------------------------------------

def build_feature_maps(img_bgr: np.ndarray):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[:, :, 0].astype(np.float32), (0, 0), 1.5)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)
    chroma = cv2.GaussianBlur(np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2), (0, 0), 1.5)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.0)
    return L, chroma, grad


def sample_channel(channel: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = channel.shape[:2]
    x = np.clip(pts[:, 0].astype(np.float32), 0, w - 1).reshape(-1, 1)
    y = np.clip(pts[:, 1].astype(np.float32), 0, h - 1).reshape(-1, 1)
    return cv2.remap(channel, x, y, cv2.INTER_LINEAR).reshape(-1)


def refine_side_on_original(L, chroma, grad, p0, p1, center, band_px):
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 5:
        return None
    d = d / length
    n = np.array([-d[1], d[0]], dtype=np.float32)
    if np.dot(center - (p0 + p1) * 0.5, n) < 0:
        n *= -1  # normal points inward

    sample_count = max(100, int(length / 8))
    alphas = np.linspace(0.08, 0.92, sample_count).astype(np.float32)
    base_mid = (p0 + p1) * 0.5

    best = None
    for angle_deg in np.linspace(-4.0, 4.0, 17):
        r = np.deg2rad(angle_deg)
        ca, sa = np.cos(r), np.sin(r)
        d2 = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]], dtype=np.float32)
        n2 = np.array([-d2[1], d2[0]], dtype=np.float32)
        if np.dot(center - base_mid, n2) < 0:
            n2 *= -1
        for off in np.linspace(-band_px, band_px, 31):
            mid = base_mid + n2 * float(off)
            start = mid - d2 * (length / 2)
            pts = start[None, :] + (alphas[:, None] * length) * d2[None, :]
            edge_pts = pts
            inside_pts = pts + n2[None, :] * 10.0
            outside_pts = pts - n2[None, :] * 14.0

            g = sample_channel(grad, edge_pts)
            l_in = sample_channel(L, inside_pts)
            l_out = sample_channel(L, outside_pts)
            c_in = sample_channel(chroma, inside_pts)
            c_out = sample_channel(chroma, outside_pts)

            brightness = np.median(l_in - l_out)
            chroma_gain = np.median(c_out - c_in)
            edge = np.median(g)
            support = np.mean((l_in > l_out + 1.5).astype(np.float32))
            score = edge * 0.7 + brightness * 0.8 + chroma_gain * 0.25 + support * 20.0
            if best is None or score > best[0]:
                best = (score, d2.copy(), mid.copy())

    if best is None:
        return None
    _, d_best, mid_best = best
    return d_best, mid_best


def intersect_lines(line_a, line_b):
    d1, p1 = line_a
    d2, p2 = line_b
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float32)
    b = p2 - p1
    try:
        t, _ = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return p1 + t * d1


def refine_quad_on_original(img_bgr: np.ndarray, quad: np.ndarray, debug_dir: Path = None) -> np.ndarray:
    """Joint constrained refinement on the original image.

    Keep opposite sides parallel and only search small offsets/rotations around the
    rough quad from Nano Banana. This is much more stable than refining each side
    independently.
    """
    L, chroma, grad = build_feature_maps(img_bgr)
    q = order_points(quad)
    tl, tr, br, bl = q

    top_dir = tr - tl
    left_dir = bl - tl
    top_len = float(np.linalg.norm(top_dir))
    left_len = float(np.linalg.norm(left_dir))
    if top_len < 10 or left_len < 10:
        return q
    top_dir /= top_len
    left_dir /= left_len
    right_dir = left_dir.copy()
    bottom_dir = top_dir.copy()

    n_top = np.array([-top_dir[1], top_dir[0]], dtype=np.float32)
    n_left = np.array([-left_dir[1], left_dir[0]], dtype=np.float32)
    center = q.mean(axis=0)
    if np.dot(center - (tl + tr) * 0.5, n_top) < 0:
        n_top *= -1
    if np.dot(center - (tl + bl) * 0.5, n_left) < 0:
        n_left *= -1

    top_mid0 = (tl + tr) * 0.5
    bot_mid0 = (bl + br) * 0.5
    left_mid0 = (tl + bl) * 0.5
    right_mid0 = (tr + br) * 0.5

    def score_line(mid, d, inward_normal, length):
        alphas = np.linspace(0.08, 0.92, max(100, int(length / 8))).astype(np.float32)
        start = mid - d * (length / 2)
        pts = start[None, :] + (alphas[:, None] * length) * d[None, :]
        edge_pts = pts
        inside_pts = pts + inward_normal[None, :] * 10.0
        outside_pts = pts - inward_normal[None, :] * 14.0
        g = sample_channel(grad, edge_pts)
        l_in = sample_channel(L, inside_pts)
        l_out = sample_channel(L, outside_pts)
        c_in = sample_channel(chroma, inside_pts)
        c_out = sample_channel(chroma, outside_pts)
        brightness = np.median(l_in - l_out)
        chroma_gain = np.median(c_out - c_in)
        edge = np.median(g)
        support = np.mean((l_in > l_out + 1.5).astype(np.float32))
        return edge * 0.8 + brightness * 1.0 + chroma_gain * 0.3 + support * 18.0

    best = None
    top_band = max(8.0, top_len / 80.0)
    left_band = max(8.0, left_len / 80.0)
    top_scores_cache = {}
    left_scores_cache = {}
    for ang_top in np.linspace(-1.5, 1.5, 5):
        rt = np.deg2rad(ang_top)
        c1, s1 = np.cos(rt), np.sin(rt)
        d_top = np.array([c1 * top_dir[0] - s1 * top_dir[1], s1 * top_dir[0] + c1 * top_dir[1]], dtype=np.float32)
        nT = np.array([-d_top[1], d_top[0]], dtype=np.float32)
        if np.dot(center - top_mid0, nT) < 0:
            nT *= -1
        d_bot = d_top.copy()
        nB = nT.copy()
        for ang_left in np.linspace(-1.5, 1.5, 5):
            rl = np.deg2rad(ang_left)
            c2, s2 = np.cos(rl), np.sin(rl)
            d_left = np.array([c2 * left_dir[0] - s2 * left_dir[1], s2 * left_dir[0] + c2 * left_dir[1]], dtype=np.float32)
            nL = np.array([-d_left[1], d_left[0]], dtype=np.float32)
            if np.dot(center - left_mid0, nL) < 0:
                nL *= -1
            d_right = d_left.copy()
            nR = nL.copy()

            top_offsets = np.linspace(-top_band, top_band, 7)
            bot_offsets = np.linspace(-top_band, top_band, 7)
            left_offsets = np.linspace(-left_band, left_band, 7)
            right_offsets = np.linspace(-left_band, left_band, 7)

            top_cache = {}
            bot_cache = {}
            left_cache = {}
            right_cache = {}
            for off_top in top_offsets:
                top_mid = top_mid0 + nT * float(off_top)
                for off_bot in np.linspace(-top_band, top_band, 15):
                    bot_mid = bot_mid0 + nB * float(off_bot)
                    if off_top not in top_cache:
                        top_cache[off_top] = score_line(top_mid, d_top, nT, top_len)
                    s_top = top_cache[off_top]
                    if off_bot not in bot_cache:
                        bot_cache[off_bot] = score_line(bot_mid, d_bot, -nB, top_len)
                    s_bot = bot_cache[off_bot]
                    for off_left in left_offsets:
                        left_mid = left_mid0 + nL * float(off_left)
                        for off_right in right_offsets:
                            right_mid = right_mid0 + nR * float(off_right)
                            if off_left not in left_cache:
                                left_cache[off_left] = score_line(left_mid, d_left, nL, left_len)
                            s_left = left_cache[off_left]
                            if off_right not in right_cache:
                                right_cache[off_right] = score_line(right_mid, d_right, -nR, left_len)
                            s_right = right_cache[off_right]
                            total = s_top + s_bot + s_left + s_right
                            if best is None or total > best[0]:
                                best = (total, (d_top, top_mid), (d_right, right_mid), (d_bot, bot_mid), (d_left, left_mid))

    if best is None:
        return q
    _, top_line, right_line, bot_line, left_line = best
    refined = [
        intersect_lines(left_line, top_line),
        intersect_lines(top_line, right_line),
        intersect_lines(right_line, bot_line),
        intersect_lines(bot_line, left_line),
    ]
    if any(p is None for p in refined):
        return q
    refined = order_points(np.array(refined, dtype=np.float32))

    if debug_dir:
        dbg = img_bgr.copy()
        cv2.polylines(dbg, [np.round(q).astype(np.int32)], True, (0, 0, 255), 3)
        for line, color in zip([top_line, right_line, bot_line, left_line], [(255,0,0),(0,255,255),(0,255,0),(255,0,255)]):
            d, p0 = line
            p1 = p0 - d * 2500
            p2 = p0 + d * 2500
            cv2.line(dbg, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), color, 2)
        cv2.polylines(dbg, [np.round(refined).astype(np.int32)], True, (255,255,255), 4)
        cv2.imwrite(str(debug_dir / "debug_original_refine.jpg"), dbg)

    return refined


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_backend(name: str, small: np.ndarray, debug_dir: Path = None):
    print(f"   🔌 Backend: {name}")
    cache_path = debug_dir / f"debug_isolated_{name}.png" if debug_dir else None
    if cache_path and cache_path.exists():
        isolated = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
    else:
        if name == "nano":
            isolated = isolate_with_nano_banana(small)
        elif name == "rembg":
            isolated = isolate_with_rembg(small)
        else:
            raise ValueError(name)
        if cache_path:
            cv2.imwrite(str(cache_path), isolated)

    if isolated.shape[:2] != small.shape[:2]:
        isolated = cv2.resize(isolated, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_LINEAR)

    drift = None
    if name == "nano":
        drift = estimate_geometry_drift(small, isolated, debug_dir=debug_dir)
        print(f"      drift: inliers={drift['inliers']} rmse={drift['rmse']:.2f}px")

    mask = mask_from_isolated(isolated, name)

    quad = fit_quad(mask)
    if quad is None:
        return None
    quad = maybe_correct_near_frontal_left_edge(mask, quad)
    mask_score = quad_iou_score(mask, quad)
    edge_score = quad_edge_support_score(small, quad)
    score = mask_score + 0.01 * edge_score
    print(f"      quad/mask IoU: {mask_score:.4f}  edge_score={edge_score:.2f}  perspective_strength={perspective_strength(quad):.3f}")

    if debug_dir:
        prefix = f"debug_{name}"
        cv2.imwrite(str(debug_dir / f"{prefix}_isolated.png"), isolated)
        cv2.imwrite(str(debug_dir / f"{prefix}_mask.png"), mask * 255)
        dbg = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
        cv2.polylines(dbg, [np.round(quad).astype(np.int32)], True, (0, 0, 255), 2)
        for p in quad:
            cv2.circle(dbg, tuple(np.round(p).astype(int)), 5, (0, 255, 255), -1)
        cv2.imwrite(str(debug_dir / f"{prefix}_fit.png"), dbg)

    return {"name": name, "isolated": isolated, "mask": mask, "quad": quad, "score": score, "mask_score": mask_score, "edge_score": edge_score, "drift": drift}


def perspective_strength(quad: np.ndarray) -> float:
    quad = order_points(quad)
    tl, tr, br, bl = quad
    top = np.linalg.norm(tr - tl)
    bot = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    width_skew = abs(top - bot) / max(top, bot, 1e-6)
    height_skew = abs(left - right) / max(left, right, 1e-6)
    return float(max(width_skew, height_skew))


def quad_edge_support_score(img_bgr: np.ndarray, quad: np.ndarray) -> float:
    """Score a quad by how well its 4 sides align to real edges in the original image.

    This helps reject backends whose mask/quad pair is self-consistent but geometrically wrong.
    """
    q = order_points(quad)
    L, chroma, grad = build_feature_maps(img_bgr)
    center = q.mean(axis=0)
    score = 0.0
    lengths = []
    for p0, p1 in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]:
        res = refine_side_on_original(L, chroma, grad, p0, p1, center, band_px=max(10.0, np.linalg.norm(p1 - p0) / 60.0))
        if res is None:
            continue
        d_best, mid_best = res
        length = float(np.linalg.norm(p1 - p0))
        lengths.append(length)
        n = np.array([-d_best[1], d_best[0]], dtype=np.float32)
        if np.dot(center - mid_best, n) < 0:
            n *= -1
        alphas = np.linspace(0.08, 0.92, max(80, int(length / 8))).astype(np.float32)
        start = mid_best - d_best * (length / 2)
        pts = start[None, :] + (alphas[:, None] * length) * d_best[None, :]
        edge_pts = pts
        inside_pts = pts + n[None, :] * 10.0
        outside_pts = pts - n[None, :] * 14.0
        g = sample_channel(grad, edge_pts)
        l_in = sample_channel(L, inside_pts)
        l_out = sample_channel(L, outside_pts)
        c_in = sample_channel(chroma, inside_pts)
        c_out = sample_channel(chroma, outside_pts)
        brightness = float(np.median(l_in - l_out))
        chroma_gain = float(np.median(c_out - c_in))
        edge = float(np.median(g))
        support = float(np.mean((l_in > l_out + 1.5).astype(np.float32)))
        score += edge * 0.7 + brightness * 0.8 + chroma_gain * 0.25 + support * 20.0

    area = abs(cv2.contourArea(q.astype(np.float32)))
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    area_ratio = area / max(img_area, 1)
    # Prefer plausible page-sized quads, but softly.
    score -= 25.0 * abs(area_ratio - 0.55)
    # Slight preference for longer, stable edges.
    if lengths:
        score += 0.002 * float(np.mean(lengths))
    return float(score)


def maybe_correct_near_frontal_left_edge(mask: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Small, local correction for near-frontal pages.

    Problem case: the receipt edge can pull the fitted left side too far right.
    We keep top/right/bottom mostly intact and only nudge the left side using
    outer-mask evidence near the top and bottom of the page.
    """
    quad = order_points(quad)
    if perspective_strength(quad) >= 0.12:
        return quad
    tl, tr, br, bl = [p.copy() for p in quad]
    h, w = mask.shape[:2]

    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return quad

    # Probe near the page top and bottom to find the outermost white pixel.
    y_top = int(np.clip(tl[1] + 0.05 * (bl[1] - tl[1]), 0, h - 1))
    y_bot = int(np.clip(bl[1] - 0.05 * (bl[1] - tl[1]), 0, h - 1))
    band = max(2, int(0.01 * h))

    def leftmost_x(yc):
        y1 = max(0, yc - band)
        y2 = min(h, yc + band + 1)
        xs = []
        for y in range(y1, y2):
            row = np.where(mask[y] > 0)[0]
            if len(row):
                xs.append(row.min())
        return None if not xs else float(np.median(xs))

    x_top = leftmost_x(y_top)
    x_bot = leftmost_x(y_bot)
    if x_top is None or x_bot is None:
        return quad

    # Only apply if this would move the left edge meaningfully outward.
    current_top_x = tl[0]
    current_bot_x = bl[0]
    if x_top >= current_top_x - 5 and x_bot >= current_bot_x - 5:
        return quad

    corrected = quad.copy()
    corrected[0, 0] = min(current_top_x, x_top)
    corrected[3, 0] = min(current_bot_x, x_bot)
    return order_points(corrected)


def detect_document(img: np.ndarray, debug_dir: Path = None, backend: str = "auto") -> np.ndarray:
    h, w = img.shape[:2]
    max_side = 1024
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    sw, sh = int(w * scale), int(h * scale)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1 else img

    backend_names = [backend] if backend != "auto" else ["nano", "rembg"]
    candidates = []
    for name in backend_names:
        try:
            cand = run_backend(name, small, debug_dir=debug_dir)
            if cand is not None:
                # Penalize Nano Banana if geometry drift is high.
                if name == "nano" and cand["drift"] is not None and not cand["drift"]["ok"]:
                    cand["score"] -= 0.25
                candidates.append(cand)
        except Exception as e:
            print(f"      {name} failed: {e}")

    if not candidates:
        raise RuntimeError("All backends failed")

    best = max(candidates, key=lambda c: c["score"])
    print(f"   ✅ Chosen backend: {best['name']} (score={best['score']:.4f})")

    quad = best["quad"] * (1.0 / scale if scale < 1 else 1.0)
    return order_points(quad)


# ---------------------------------------------------------------------------
# Perspective warp with true aspect ratio recovery
# ---------------------------------------------------------------------------

def _line_intersect_2d(p1, p2, p3, p4):
    """Intersection of line p1-p2 and p3-p4. Returns None if parallel."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float64)


def recover_aspect_ratio(corners: np.ndarray, img_shape) -> float:
    """Recover the true width/height ratio of the document from its projected corners.

    Method: single-view geometry (Criminisi / Zhang).
    1. Find vanishing points of horizontal and vertical edges.
    2. Recover focal length from the orthogonality constraint.
    3. Lift the 4 corners to 3D and compute true edge lengths.
    Falls back to naive pixel ratio if the geometry is degenerate.
    """
    tl, tr, br, bl = [c.astype(np.float64) for c in corners]
    h, w = img_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Vanishing point of horizontal edges (top: TL→TR, bottom: BL→BR)
    v1 = _line_intersect_2d(tl, tr, bl, br)
    # Vanishing point of vertical edges (left: TL→BL, right: TR→BR)
    v2 = _line_intersect_2d(tl, bl, tr, br)

    def naive_ratio():
        W = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        H = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        return W / H if H > 0 else 1.0

    if v1 is None or v2 is None:
        return naive_ratio()

    # Focal length from orthogonality: f² = -(V1-c)·(V2-c)
    v1c = v1 - np.array([cx, cy])
    v2c = v2 - np.array([cx, cy])
    f2 = -np.dot(v1c, v2c)
    if f2 <= 0:
        return naive_ratio()
    f = np.sqrt(f2)

    # Lift image points to 3D direction vectors via K⁻¹
    def lift(p):
        return np.array([(p[0] - cx) / f, (p[1] - cy) / f, 1.0])

    # Plane normal = cross product of the two vanishing directions
    d1 = lift(v1); d1 /= np.linalg.norm(d1)
    d2 = lift(v2); d2 /= np.linalg.norm(d2)
    N = np.cross(d1, d2)
    norm_n = np.linalg.norm(N)
    if norm_n < 1e-8:
        return naive_ratio()
    N /= norm_n

    # Depth of each corner: λᵢ = 1 / (N · dirᵢ)  (up to global scale)
    def depth(p):
        d = lift(p)
        nd = np.dot(N, d)
        return None if abs(nd) < 1e-8 else 1.0 / nd

    lam = [depth(p) for p in (tl, tr, br, bl)]
    if any(x is None or x <= 0 for x in lam):
        return naive_ratio()

    # 3D positions (up to the shared scale factor, which cancels in ratios)
    def pos3d(p, lam):
        return lam * lift(p)

    tl3 = pos3d(tl, lam[0])
    tr3 = pos3d(tr, lam[1])
    br3 = pos3d(br, lam[2])
    bl3 = pos3d(bl, lam[3])

    W3 = (np.linalg.norm(tr3 - tl3) + np.linalg.norm(br3 - bl3)) / 2
    H3 = (np.linalg.norm(bl3 - tl3) + np.linalg.norm(br3 - tr3)) / 2
    if H3 < 1e-6:
        return naive_ratio()
    return W3 / H3


def warp_document(img: np.ndarray, corners: np.ndarray,
                  aspect_ratio: float = None) -> np.ndarray:
    """Perspective-correct the document.

    aspect_ratio: width/height of the output.
      None  → auto-recover from the projected corners (default, recommended)
      float → force a specific ratio (e.g. 1/1.4142 for A4 portrait)
    """
    corners = order_points(corners)
    if aspect_ratio is None:
        aspect_ratio = recover_aspect_ratio(corners, img.shape)

    # Choose output size: base it on the longer projected diagonal, capped at 3000px
    diag = max(
        np.linalg.norm(corners[2] - corners[0]),
        np.linalg.norm(corners[3] - corners[1]),
    )
    long_side = min(int(diag), 3000)
    if aspect_ratio >= 1.0:  # landscape
        W, H = long_side, int(round(long_side / aspect_ratio))
    else:  # portrait
        H, W = long_side, int(round(long_side * aspect_ratio))

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


# ---------------------------------------------------------------------------
# Step 3: scan enhancement
# ---------------------------------------------------------------------------

def _flatten_background(L: np.ndarray) -> np.ndarray:
    """Remove shadows/uneven lighting without amplifying faint marks.

    Divides by a blurred background estimate. No percentile stretching,
    so we don't drag faint bleed-through up into visible territory.
    """
    h, w = L.shape[:2]
    sigma = max(h, w) * 0.06
    bg = cv2.GaussianBlur(L, (0, 0), sigma)
    # Use a high percentile of bg as the reference white level.
    white_ref = float(np.percentile(bg, 95))
    white_ref = max(white_ref, 1.0)
    flat = np.clip(L / bg * white_ref, 0, 255)
    return flat


def _white_clamp(L: np.ndarray, low: int = 200, high: int = 240) -> np.ndarray:
    """Smoothly lift near-white values to pure white.

    This is what physical scanners do: the paper itself is treated as
    the white point so faint bleed-through from the next page disappears
    without blowing out actual text or stamps.
    """
    x = L.astype(np.float32)
    # Below `low`: untouched. Between low..high: ramp to 255. Above high: 255.
    ramp = np.clip((x - low) / max(high - low, 1), 0, 1)
    out = x * (1 - ramp) + 255.0 * ramp
    return np.clip(out, 0, 255)


def enhance_scan(img: np.ndarray, mode: str = "gray") -> np.ndarray:
    """Turn a perspective-corrected photo into a clean scan.

    Modes:
      'gray'    – faithful grayscale scan, bleed-through suppressed (default)
      'archive' – slightly crisper grayscale, still bleed-through safe
      'color'   – color preserved, shadows removed
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # 1. Flatten lighting (no percentile stretch, keeps faint marks faint).
    L_flat = _flatten_background(L)

    # 2. Mild denoise.
    L8 = L_flat.astype(np.uint8)
    L_denoised = cv2.fastNlMeansDenoising(L8, None, h=4, templateWindowSize=7, searchWindowSize=21)

    # 3. Clamp near-white to white so the next-page bleed-through disappears.
    if mode == "archive":
        L_clean = _white_clamp(L_denoised, low=195, high=235).astype(np.uint8)
    else:
        L_clean = _white_clamp(L_denoised, low=205, high=240).astype(np.uint8)

    def unsharp(x, sigma=1.0, amount=0.4):
        blur = cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)
        sharp = np.clip(x.astype(np.float32) * (1.0 + amount) - blur * amount, 0, 255)
        return sharp.astype(np.uint8)

    if mode == "gray":
        out = unsharp(L_clean, sigma=1.0, amount=0.3)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if mode == "archive":
        out = unsharp(L_clean, sigma=1.0, amount=0.6)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if mode == "color":
        L_sharp = unsharp(L_clean, sigma=1.0, amount=0.3).astype(np.float32)
        lab_out = lab.copy()
        lab_out[:, :, 0] = L_sharp
        # Desaturate where the result is close to white so bleed-through
        # doesn't reappear in color.
        weight = np.clip((L_sharp - 210) / 30.0, 0, 1)[:, :, None]
        lab_out[:, :, 1:] = lab[:, :, 1:] * (1 - weight) + 128.0 * weight
        out = cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Visualization / CLI
# ---------------------------------------------------------------------------

def draw_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    vis = img.copy()
    pts = np.round(corners).astype(int)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 6)
    for p in pts:
        cv2.circle(vis, tuple(p), 16, (0, 0, 255), -1)
    return vis


def slugify_name(name: str) -> str:
    name = Path(name).stem.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name or "image"


def build_step_path(base_dir: Path, step_num: int, step_name: str, image_slug: str, ext: str = ".jpg") -> Path:
    step_slug = re.sub(r"[^a-z0-9]+", "-", step_name.lower()).strip("-")
    return base_dir / f"{step_num:02d}-{step_slug}-{image_slug}{ext}"


def cmd_scan(args):
    """Handle the scan subcommand."""
    inp = Path(args.input)
    img = cv2.imread(str(inp))
    if img is None:
        print(f"Error: could not read {inp}", file=sys.stderr)
        sys.exit(1)

    debug_dir = SCRIPT_DIR if args.debug else None
    image_slug = slugify_name(inp.name)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Detecting document corners...")
    corners = detect_document(img, debug_dir=debug_dir, backend=args.backend)

    print("✅ Final corners:")
    for name, pt in zip(["top-left", "top-right", "bottom-right", "bottom-left"], corners):
        print(f"   {name}: ({pt[0]:.0f}, {pt[1]:.0f})")

    overlay_path = build_step_path(out_dir, 1, "corners", image_slug)
    cv2.imwrite(str(overlay_path), draw_corners(img, corners))
    print(f"📄 Overlay → {overlay_path}")

    warped = warp_document(img, corners)
    warp_path = build_step_path(out_dir, 2, "warped", image_slug)
    cv2.imwrite(str(warp_path), warped)
    print(f"📄 Warped  → {warp_path}")

    enhanced = enhance_scan(warped, mode=args.mode)
    enhance_path = build_step_path(out_dir, 3, f"enhanced-{args.mode}", image_slug)
    cv2.imwrite(str(enhance_path), enhanced)
    print(f"✨ Enhanced → {enhance_path} (mode={args.mode})")


def cmd_pdf(args):
    """Handle the pdf subcommand."""
    from scanario.pdf_utils import create_pdf_from_images, collect_pages
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Walk sources in order. Each source becomes one or more pages.
    # Default: files are included as-is (no reprocessing). Use --process to
    # run the scan pipeline on each file before including it.
    pages_in_order: list[Path] = []
    temp_pages: list[Path] = []

    def pick_from_dir(d: Path) -> Path | None:
        for pattern in ("03-enhanced-*.jpg", "02-warped-*.jpg", "*.jpg", "*.png"):
            hits = sorted(d.glob(pattern))
            if hits:
                return hits[0]
        return None

    for i, source in enumerate(args.sources):
        source_path = Path(source)
        if not source_path.exists():
            print(f"Warning: skipping {source} (not found)", file=sys.stderr)
            continue

        if source_path.is_dir():
            picked = pick_from_dir(source_path)
            if picked is None:
                print(f"Warning: no images in {source_path}", file=sys.stderr)
                continue
            print(f"  [{i+1}] {source_path}/ → {picked.name}")
            pages_in_order.append(picked)
            continue

        # It's a file.
        if not args.process:
            print(f"  [{i+1}] {source_path.name} (as-is)")
            pages_in_order.append(source_path)
            continue

        # --process: run the full scan pipeline.
        print(f"  [{i+1}] {source_path.name} (processing...)")
        img = cv2.imread(str(source_path))
        if img is None:
            print(f"    Error: could not read {source_path}", file=sys.stderr)
            continue
        debug_dir = SCRIPT_DIR if args.debug else None
        corners = detect_document(img, debug_dir=debug_dir, backend=args.backend)
        if corners is None:
            print(f"    Error: could not detect corners", file=sys.stderr)
            continue
        warped = warp_document(img, corners)
        enhanced = enhance_scan(warped, mode=args.mode)
        temp_path = output_path.parent / f"_page_{i:03d}.jpg"
        cv2.imwrite(str(temp_path), enhanced)
        pages_in_order.append(temp_path)
        temp_pages.append(temp_path)

    if not pages_in_order:
        print("Error: no valid pages to include in PDF", file=sys.stderr)
        sys.exit(1)

    print(f"\nCreating PDF with {len(pages_in_order)} pages...")
    create_pdf_from_images(pages_in_order, output_path, dpi=args.dpi)
    print(f"✅ PDF created: {output_path}")

    for temp_path in temp_pages:
        temp_path.unlink(missing_ok=True)


def cmd_auth(args):
    """Manage API keys stored in SCANARIO_DATA_DIR/auth-keys.json."""
    try:
        from scanario import auth
    except ImportError as exc:
        print(f"Error: could not load auth module: {exc}", file=sys.stderr)
        sys.exit(1)

    action = args.auth_action
    if action == "create":
        try:
            key = auth.create_key(label=args.label or "")
        except Exception as exc:
            print(f"Error: could not write auth file ({exc}).", file=sys.stderr)
            print("Make sure SCANARIO_DATA_DIR is writable.", file=sys.stderr)
            sys.exit(1)
        print("✅ New API key created. Save it now – it will NOT be shown again:\n")
        print(f"   {key}\n")
        print("Send it on every request as either:")
        print("  X-API-Key: <key>")
        print("  Authorization: Bearer <key>")
        return

    if action == "list":
        keys = auth.list_keys()
        if not keys:
            print("No API keys yet. Create one with: scanario auth create")
            return
        print(f"{'PREFIX':<20}  {'CREATED':<30}  LABEL")
        for k in keys:
            print(f"{k.prefix:<20}  {k.created_at:<30}  {k.label}")
        return

    if action == "revoke":
        n = auth.revoke_by_prefix(args.prefix)
        print(f"Revoked {n} key(s).")
        return

    print("Unknown auth action", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="scanario – document corner detector and PDF creator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a single document")
    scan_parser.add_argument("input", help="Input image path")
    scan_parser.add_argument("--out-dir", required=True, help="Output directory; saves numbered step files like 01-corners-bad-perspective.jpg")
    scan_parser.add_argument("--mode", choices=["gray", "archive", "color"], default="gray", help="Enhancement mode")
    scan_parser.add_argument("--backend", choices=["auto", "nano", "rembg"], default="auto", help="Isolation backend")
    scan_parser.add_argument("--debug", action="store_true", help="Save intermediate debug images")

    # PDF command
    pdf_parser = subparsers.add_parser("pdf", help="Create PDF from multiple images")
    pdf_parser.add_argument("output", help="Output PDF path")
    pdf_parser.add_argument("sources", nargs="+", help="Input images or result directories to include as pages")
    pdf_parser.add_argument("--mode", choices=["gray", "archive", "color"], default="gray", help="Enhancement mode for new images")
    pdf_parser.add_argument("--backend", choices=["auto", "nano", "rembg"], default="auto", help="Isolation backend for new images")
    pdf_parser.add_argument("--debug", action="store_true", help="Save intermediate debug images")
    pdf_parser.add_argument("--dpi", type=int, default=300, help="PDF resolution in DPI")
    pdf_parser.add_argument("--process", action="store_true", help="Re-run the scan pipeline on each input file (default: include files as-is)")

    # Auth command
    auth_parser = subparsers.add_parser("auth", help="Manage API keys")
    auth_sub = auth_parser.add_subparsers(dest="auth_action", required=True)

    auth_create = auth_sub.add_parser("create", help="Create a new API key")
    auth_create.add_argument("--label", help="Optional label for this key")

    auth_sub.add_parser("list", help="List stored API keys (prefixes only)")

    auth_revoke = auth_sub.add_parser("revoke", help="Revoke keys by prefix")
    auth_revoke.add_argument("prefix", help="Full key or any prefix (e.g. 'sk_abc123')")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "pdf":
        cmd_pdf(args)
    elif args.command == "auth":
        cmd_auth(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
