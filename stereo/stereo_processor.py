# """
# StereoProcessor
# ---------------
# Encapsulates the stereo-vision pipeline:

# Three rectification modes are supported:

#   * "none"         - Skip rectification entirely. Use the raw images
#                      directly. This is the right choice when the cameras
#                      are already aligned (simulation environments like
#                      Gazebo, or a properly calibrated rig publishing
#                      pre-rectified images).
#   * "uncalibrated" - Original project's method: SIFT + BFMatcher +
#                      RANSAC 8-point -> F -> cv2.stereoRectifyUncalibrated
#                      -> H1, H2. Computed once on the first frame.
#                      Geometrically correct for epipolar matching but
#                      can warp images significantly, especially when
#                      the cameras are actually already aligned.
#   * "calibrated"   - Use the CameraInfo's projection matrix P (which
#                      ROS publishes pre-rectified) via cv2.initUndistort-
#                      RectifyMap with K, D, R, P. Needs a properly
#                      calibrated stereo pair.

# Distance modes (what gets returned to the user):

#   * "z"      - Cartesian Z component: perpendicular distance from the
#                image plane. What a "depth map" classically means.
#   * "radial" - Euclidean distance from the camera's optical centre to
#                the 3D point: sqrt(X^2 + Y^2 + Z^2). More physically
#                meaningful when the user points at off-centre pixels
#                on a flat surface; the corner of a flat wall really IS
#                further from the lens than its centre, and radial
#                reflects that. Default for this build.

# Per-frame pipeline (after rectification, if any):
#   - StereoSGBM disparity (left)
#   - Right-matcher disparity (only if WLS enabled)
#   - WLS post-filter (weighted least squares) - fills holes, sharpens
#     edges, suppresses streaking. Requires opencv-contrib-python
#     (cv2.ximgproc). Disabled gracefully if the module is missing.
#   - Optional confidence-based masking: WLS produces a per-pixel
#     confidence map; pixels below `wls_conf_threshold` (0-255) are
#     marked invalid so they are excluded from depth and ROI median.
#   - Optional median post-filter for any leftover speckles.
#   - depth_mm Z = (baseline_mm * f_px) / disparity_px
#   - if distance_mode == "radial":  R(u,v) = Z * sqrt(1 + (u-cx)^2/fx^2
#                                                        + (v-cy)^2/fy^2)
# """
# import numpy as np
# import cv2

# from .utils.fundamental_matrix import get_inliers
# from .utils.misc_utils import sift_features_to_array


# VALID_MODES = ("none", "uncalibrated", "calibrated")
# VALID_DISTANCE_MODES = ("z", "radial")

# # WLS post-filter and right-matcher live in opencv-contrib-python.
# # If the user only has the base opencv-python package, gracefully
# # fall back to the un-filtered SGBM output.
# _HAS_XIMGPROC = hasattr(cv2, "ximgproc")


# class StereoProcessor:
#     def __init__(self, baseline_mm=80.0,
#                  sgbm_num_disparities=128,
#                  sgbm_block_size=5,
#                  max_features=300,
#                  rectification_mode="none",
#                  distance_mode="radial",
#                  median_filter_size=5,
#                  use_wls_filter=True,
#                  wls_lambda=8000.0,
#                  wls_sigma=1.5,
#                  wls_conf_threshold=0):
#         self.baseline_mm = float(baseline_mm)
#         self.max_features = int(max_features)

#         if rectification_mode not in VALID_MODES:
#             raise ValueError(
#                 f"rectification_mode must be one of {VALID_MODES}, "
#                 f"got '{rectification_mode}'")
#         self.mode = rectification_mode

#         if distance_mode not in VALID_DISTANCE_MODES:
#             raise ValueError(
#                 f"distance_mode must be one of {VALID_DISTANCE_MODES}, "
#                 f"got '{distance_mode}'")
#         self.distance_mode = distance_mode

#         # Median filter on disparity. Cheap noise suppression applied
#         # AFTER WLS (if any).  Capped at 5 because cv2.medianBlur on
#         # CV_32F only supports ksize in {3, 5}; larger values throw.
#         # Set to 0 or 1 to disable.
#         mfs = int(median_filter_size)
#         if mfs <= 1:
#             mfs = 0
#         else:
#             if mfs % 2 == 0:
#                 mfs += 1
#             mfs = min(mfs, 5)
#         self.median_filter_size = mfs

#         # ---- WLS configuration ----
#         self.wls_conf_threshold = int(wls_conf_threshold)
#         self._wls_requested = bool(use_wls_filter)
#         self.use_wls = self._wls_requested and _HAS_XIMGPROC
#         # True when the user asked for WLS but cv2.ximgproc isn't installed.
#         # The node uses this to print a warning.
#         self.wls_requested_but_unavailable = (
#             self._wls_requested and not _HAS_XIMGPROC)

#         # Rectification state (used by "uncalibrated" mode)
#         self.H1 = None
#         self.H2 = None
#         self.F = None
#         # Maps used by "calibrated" mode
#         self.map1_l = self.map2_l = None
#         self.map1_r = self.map2_r = None
#         # Generic "are we ready to rectify?" flag
#         self.calibrated = (self.mode == "none")  # "none" is always ready

#         # Camera intrinsics (filled from CameraInfo)
#         self.K1 = None
#         self.K2 = None
#         self.D1 = None
#         self.D2 = None
#         self.R1_rect = None
#         self.R2_rect = None
#         self.P1 = None
#         self.P2 = None
#         self.image_size = None
#         self.focal_length = None  # average of fx1, fx2

#         # Cached radial scale factor map (lazy-built per image size).
#         # radial = Z * radial_scale_map ;  shape = (h, w), dtype float32.
#         self._radial_scale = None
#         self._radial_scale_for_size = None  # (w, h) tuple

#         # ---- SGBM ----
#         if sgbm_num_disparities % 16 != 0:
#             sgbm_num_disparities = ((sgbm_num_disparities // 16) + 1) * 16
#         bs = sgbm_block_size
#         self.sgbm = cv2.StereoSGBM_create(
#             minDisparity=0,
#             numDisparities=sgbm_num_disparities,
#             blockSize=bs,
#             P1=8 * 3 * bs * bs,
#             P2=32 * 3 * bs * bs,
#             disp12MaxDiff=1,
#             uniquenessRatio=10,
#             speckleWindowSize=100,
#             speckleRange=32,
#             preFilterCap=63,
#             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
#         )

#         # ---- Right-matcher + WLS post-filter (if available) ----
#         self.right_matcher = None
#         self.wls = None
#         if self.use_wls:
#             # createRightMatcher takes the left matcher and produces a
#             # configuration suitable for matching right->left along the
#             # same epipolar lines. Together they give the WLS filter
#             # the left-right consistency it needs.
#             self.right_matcher = cv2.ximgproc.createRightMatcher(self.sgbm)
#             self.wls = cv2.ximgproc.createDisparityWLSFilter(
#                 matcher_left=self.sgbm)
#             self.wls.setLambda(float(wls_lambda))
#             self.wls.setSigmaColor(float(wls_sigma))

#     # ------------------------------------------------------------------ #
#     #  Read-only flags                                                   #
#     # ------------------------------------------------------------------ #
#     @property
#     def wls_enabled(self):
#         return self.use_wls and self.wls is not None

#     # ------------------------------------------------------------------ #
#     #  Camera info                                                       #
#     # ------------------------------------------------------------------ #
#     def set_camera_info(self, K1, K2,
#                         D1=None, D2=None,
#                         R1=None, R2=None,
#                         P1=None, P2=None,
#                         image_size=None):
#         self.K1 = np.asarray(K1, dtype=np.float64).reshape(3, 3)
#         self.K2 = np.asarray(K2, dtype=np.float64).reshape(3, 3)
#         if D1 is not None:
#             self.D1 = np.asarray(D1, dtype=np.float64).ravel()
#         if D2 is not None:
#             self.D2 = np.asarray(D2, dtype=np.float64).ravel()
#         if R1 is not None:
#             self.R1_rect = np.asarray(R1, dtype=np.float64).reshape(3, 3)
#         if R2 is not None:
#             self.R2_rect = np.asarray(R2, dtype=np.float64).reshape(3, 3)
#         if P1 is not None:
#             self.P1 = np.asarray(P1, dtype=np.float64).reshape(3, 4)
#         if P2 is not None:
#             self.P2 = np.asarray(P2, dtype=np.float64).reshape(3, 4)
#         if image_size is not None:
#             self.image_size = (int(image_size[0]), int(image_size[1]))

#         # Focal length: in calibrated mode use rectified fx from P.
#         if self.mode == "calibrated" and self.P1 is not None and self.P2 is not None:
#             self.focal_length = 0.5 * (self.P1[0, 0] + self.P2[0, 0])
#         else:
#             self.focal_length = 0.5 * (self.K1[0, 0] + self.K2[0, 0])

#         if self.mode == "calibrated":
#             self._build_calibrated_maps()

#         # Camera intrinsics changed -> radial scale map needs rebuilding.
#         self._radial_scale = None
#         self._radial_scale_for_size = None

#     # ------------------------------------------------------------------ #
#     #  Calibration                                                       #
#     # ------------------------------------------------------------------ #
#     def _build_calibrated_maps(self):
#         if (self.K1 is None or self.K2 is None or
#                 self.R1_rect is None or self.R2_rect is None or
#                 self.P1 is None or self.P2 is None or
#                 self.image_size is None):
#             return False
#         D1 = self.D1 if self.D1 is not None else np.zeros(5)
#         D2 = self.D2 if self.D2 is not None else np.zeros(5)
#         self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
#             self.K1, D1, self.R1_rect, self.P1,
#             self.image_size, cv2.CV_32FC1)
#         self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
#             self.K2, D2, self.R2_rect, self.P2,
#             self.image_size, cv2.CV_32FC1)
#         self.calibrated = True
#         return True

#     def calibrate(self, img_left, img_right):
#         if self.mode == "none":
#             self.calibrated = True
#             return True

#         if self.mode == "calibrated":
#             return self._build_calibrated_maps()

#         # "uncalibrated"
#         if img_left is None or img_right is None:
#             return False
#         if img_left.shape[:2] != img_right.shape[:2]:
#             return False

#         gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) \
#             if img_left.ndim == 3 else img_left
#         gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) \
#             if img_right.ndim == 3 else img_right

#         sift = cv2.SIFT_create()
#         kp1, des1 = sift.detectAndCompute(gray_l, None)
#         kp2, des2 = sift.detectAndCompute(gray_r, None)
#         if des1 is None or des2 is None:
#             return False
#         if len(kp1) < 8 or len(kp2) < 8:
#             return False

#         bf = cv2.BFMatcher()
#         matches = bf.match(des1, des2)
#         matches = sorted(matches, key=lambda m: m.distance)
#         chosen = matches[: self.max_features]
#         if len(chosen) < 8:
#             return False

#         pairs = sift_features_to_array(chosen, kp1, kp2)

#         F_best, inliers = get_inliers(pairs)
#         if F_best is None or inliers.shape[0] < 8:
#             return False

#         set1 = inliers[:, 0:2].astype(np.float32)
#         set2 = inliers[:, 2:4].astype(np.float32)

#         h, w = gray_l.shape[:2]
#         ok, H1, H2 = cv2.stereoRectifyUncalibrated(
#             set1.reshape(-1, 1, 2),
#             set2.reshape(-1, 1, 2),
#             F_best,
#             imgSize=(w, h),
#         )
#         if not ok:
#             return False

#         self.F = F_best
#         self.H1 = H1
#         self.H2 = H2
#         self.calibrated = True
#         return True

#     def reset_calibration(self):
#         if self.mode == "none":
#             return
#         self.calibrated = False
#         self.H1 = self.H2 = self.F = None
#         self.map1_l = self.map2_l = None
#         self.map1_r = self.map2_r = None

#     # ------------------------------------------------------------------ #
#     #  Per-frame pipeline                                                #
#     # ------------------------------------------------------------------ #
#     def rectify(self, img_left, img_right):
#         if self.mode == "none":
#             return img_left, img_right
#         if not self.calibrated:
#             return None, None
#         if self.mode == "uncalibrated":
#             h, w = img_left.shape[:2]
#             rect_l = cv2.warpPerspective(img_left, self.H1, (w, h))
#             rect_r = cv2.warpPerspective(img_right, self.H2, (w, h))
#             return rect_l, rect_r
#         rect_l = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
#         rect_r = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
#         return rect_l, rect_r

#     def compute_disparity(self, rect_left, rect_right):
#         g_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY) \
#             if rect_left.ndim == 3 else rect_left
#         g_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY) \
#             if rect_right.ndim == 3 else rect_right

#         # SGBM returns disparity*16 in int16 (fixed-point).  WLS expects
#         # exactly that format, so keep it as int16 until after WLS.
#         disp_left_16 = self.sgbm.compute(g_l, g_r)

#         if self.use_wls and self.wls is not None:
#             # Right matcher: same SGBM parameters but matching the right
#             # image against the left. The two disparity maps let WLS run
#             # a left-right consistency check, mark inconsistent pixels
#             # as low-confidence, then fill them using a guided edge-aware
#             # smoothing of the left image. This is what kills the
#             # "alacalı" colour-noise pattern in the raw disparity.
#             #
#             # NOTE the argument order: (g_r, g_l).
#             disp_right_16 = self.right_matcher.compute(g_r, g_l)
#             filtered_16 = self.wls.filter(
#                 disp_left_16, g_l, disparity_map_right=disp_right_16)
#             disp = filtered_16.astype(np.float32) / 16.0

#             # Optional confidence masking. WLS confidence map is in
#             # [0, 255] (higher = more reliable). Pixels under threshold
#             # are forced to "invalid" so the depth step skips them
#             # entirely. Conservative for measurement work: better to
#             # report "no valid pixel here" than to publish a wrong mm.
#             if self.wls_conf_threshold > 0:
#                 try:
#                     conf = self.wls.getConfidenceMap()
#                     if conf is not None and conf.shape == disp.shape:
#                         disp[conf < float(self.wls_conf_threshold)] = -1.0
#                 except cv2.error:
#                     pass
#         else:
#             disp = disp_left_16.astype(np.float32) / 16.0

#         # Cheap median post-filter to mop up any leftover per-pixel
#         # speckles.  Done AFTER WLS because WLS already smooths
#         # heavily; a tiny 3x3 or 5x5 here just chases isolated outliers.
#         if self.median_filter_size >= 3:
#             disp = cv2.medianBlur(disp, self.median_filter_size)

#         return disp

#     # ------------------------------------------------------------------ #
#     #  Depth / distance                                                  #
#     # ------------------------------------------------------------------ #
#     def _ensure_radial_scale(self, shape):
#         """Build (and cache) a per-pixel scale map S(u,v) such that
#         radial_distance = Z * S(u,v).

#         Derivation: for a pinhole camera, a pixel (u,v) corresponds to
#         a ray with direction (x', y', 1) in normalised coords, where
#         x' = (u-cx)/fx and y' = (v-cy)/fy.  The 3D point at depth Z is
#         (Z*x', Z*y', Z) and its Euclidean distance from the camera
#         origin is Z * sqrt(1 + x'^2 + y'^2).  So S = sqrt(1 + x'^2 + y'^2).
#         """
#         h, w = shape[:2]
#         if (self._radial_scale is not None and
#                 self._radial_scale_for_size == (w, h)):
#             return self._radial_scale

#         # Prefer rectified intrinsics from P when in calibrated mode.
#         if self.mode == "calibrated" and self.P1 is not None:
#             fx = float(self.P1[0, 0])
#             fy = float(self.P1[1, 1])
#             cx = float(self.P1[0, 2])
#             cy = float(self.P1[1, 2])
#         elif self.K1 is not None:
#             fx = float(self.K1[0, 0])
#             fy = float(self.K1[1, 1])
#             cx = float(self.K1[0, 2])
#             cy = float(self.K1[1, 2])
#         else:
#             # Fallback: assume principal point at image centre and a
#             # crude focal length. Better than nothing if CameraInfo
#             # was never received (shouldn't happen with our node).
#             fx = fy = max(w, h)
#             cx = (w - 1) * 0.5
#             cy = (h - 1) * 0.5

#         # Build x', y' grids in normalised camera coords.
#         u = np.arange(w, dtype=np.float32)
#         v = np.arange(h, dtype=np.float32)
#         uu, vv = np.meshgrid(u, v)
#         x_n = (uu - cx) / fx
#         y_n = (vv - cy) / fy
#         scale = np.sqrt(1.0 + x_n * x_n + y_n * y_n).astype(np.float32)

#         self._radial_scale = scale
#         self._radial_scale_for_size = (w, h)
#         return scale

#     def compute_depth_mm(self, disparity):
#         """Returns a per-pixel distance map in millimetres.

#         - distance_mode == "z":      Z = (baseline * f) / disparity
#         - distance_mode == "radial": R = Z * sqrt(1 + x_n^2 + y_n^2)

#         Invalid pixels (disparity too small) are returned as 0.
#         """
#         if self.focal_length is None:
#             return None

#         Z = np.zeros_like(disparity, dtype=np.float32)
#         valid = disparity > 0.5
#         Z[valid] = (self.baseline_mm * self.focal_length) / disparity[valid]

#         if self.distance_mode == "z":
#             return Z

#         # radial
#         scale = self._ensure_radial_scale(disparity.shape)
#         R = Z * scale  # invalid pixels stay 0 because Z is 0 there
#         return R

#     @staticmethod
#     def region_median_depth(depth_map, mask):
#         if depth_map is None or mask is None:
#             return None
#         sel = depth_map[(mask > 0) & (depth_map > 0)]
#         if sel.size == 0:
#             return None
#         return float(np.median(sel))





"""
StereoProcessor
---------------
Encapsulates the stereo-vision pipeline:

Three rectification modes are supported:

  * "none"         - Skip rectification entirely. Use the raw images
                     directly. This is the right choice when the cameras
                     are already aligned (simulation environments like
                     Gazebo, or a properly calibrated rig publishing
                     pre-rectified images).
  * "uncalibrated" - Original project's method: SIFT + BFMatcher +
                     RANSAC 8-point -> F -> cv2.stereoRectifyUncalibrated
                     -> H1, H2. Computed once on the first frame.
                     Geometrically correct for epipolar matching but
                     can warp images significantly, especially when
                     the cameras are actually already aligned.
  * "calibrated"   - Use the CameraInfo's projection matrix P (which
                     ROS publishes pre-rectified) via cv2.initUndistort-
                     RectifyMap with K, D, R, P. Needs a properly
                     calibrated stereo pair.

Distance modes (what gets returned to the user):

  * "z"      - Cartesian Z component: perpendicular distance from the
               image plane. What a "depth map" classically means.
  * "radial" - Euclidean distance from the camera's optical centre to
               the 3D point: sqrt(X^2 + Y^2 + Z^2). More physically
               meaningful when the user points at off-centre pixels
               on a flat surface; the corner of a flat wall really IS
               further from the lens than its centre, and radial
               reflects that. Default for this build.

Per-frame pipeline (after rectification, if any):
  - Optional CLAHE pre-processing on both grayscale views. Strongly
    recommended for scenes with glossy/specular surfaces (chairs,
    monitors, skin) and low-texture areas (white walls, plain
    clothing) - the per-tile contrast boost gives SGBM something to
    match on.
  - StereoSGBM disparity (left). Mode is configurable: `sgbm_3way`
    (fastest, can streak), `sgbm` (5-direction, balanced default),
    `hh` (full 8-direction, slowest but cleanest), `hh4`.
  - Right-matcher disparity (only if WLS enabled).
  - WLS post-filter (weighted least squares) - left-right consistency
    + edge-aware smoothing. Requires opencv-contrib-python; falls
    back gracefully.
  - Optional confidence-based masking: pixels under
    `wls_conf_threshold` (0-255) become invalid.
  - Optional median post-filter for residual speckles.
  - depth_mm Z = (baseline_mm * f_px) / disparity_px
  - if distance_mode == "radial":  R(u,v) = Z * sqrt(1 + (u-cx)^2/fx^2
                                                       + (v-cy)^2/fy^2)
"""
"""
StereoProcessor
---------------
Encapsulates the stereo-vision pipeline.

Rectification modes:
  * "none"         - skip rectification (cameras already aligned)
  * "uncalibrated" - SIFT + RANSAC F + stereoRectifyUncalibrated
  * "calibrated"   - use CameraInfo K, D, R, P via initUndistortRectifyMap

Distance modes:
  * "z"      - perpendicular depth from image plane
  * "radial" - Euclidean distance from camera optical centre

Per-frame pipeline (after rectification):
  - Optional CLAHE on both gray views (specular/texture-less surfaces).
  - StereoSGBM disparity (mode configurable).
  - Right-matcher disparity (only if WLS enabled).
  - WLS post-filter (needs opencv-contrib-python).
  - Optional confidence-based masking.
  - Optional median post-filter.
  - depth_mm = (baseline_mm * f_px) / disparity_px
  - radial:  R = Z * sqrt(1 + (u-cx)^2/fx^2 + (v-cy)^2/fy^2)

Runtime updates:
  The build_* / update_* methods rebuild the relevant internal objects
  so a parameter callback in the node can apply changes without
  recreating the whole processor.
"""
import numpy as np
import cv2

from .utils.fundamental_matrix import get_inliers
from .utils.misc_utils import sift_features_to_array


VALID_MODES = ("none", "uncalibrated", "calibrated")
VALID_DISTANCE_MODES = ("z", "radial")

_HAS_XIMGPROC = hasattr(cv2, "ximgproc")

_SGBM_MODES = {
    "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
    "sgbm_3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    "hh": cv2.STEREO_SGBM_MODE_HH,
}
if hasattr(cv2, "STEREO_SGBM_MODE_HH4"):
    _SGBM_MODES["hh4"] = cv2.STEREO_SGBM_MODE_HH4

VALID_SGBM_MODES = tuple(_SGBM_MODES.keys())


def _odd_clamp(v, lo, hi):
    v = int(v)
    if v % 2 == 0:
        v += 1
    return max(lo, min(hi, v))


def _multiple_of_16(v):
    v = int(v)
    if v % 16 != 0:
        v = ((v // 16) + 1) * 16
    return max(16, v)


class StereoProcessor:
    def __init__(self, baseline_mm=80.0,
                 sgbm_num_disparities=128,
                 sgbm_block_size=5,
                 sgbm_mode="sgbm",
                 sgbm_disp12_max_diff=5,
                 sgbm_uniqueness_ratio=10,
                 max_features=300,
                 rectification_mode="none",
                 distance_mode="radial",
                 median_filter_size=5,
                 use_wls_filter=True,
                 wls_lambda=8000.0,
                 wls_sigma=1.5,
                 wls_conf_threshold=0,
                 use_clahe=True,
                 clahe_clip_limit=2.0,
                 clahe_grid_size=8):
        self.baseline_mm = float(baseline_mm)
        self.max_features = int(max_features)

        if rectification_mode not in VALID_MODES:
            raise ValueError(
                f"rectification_mode must be one of {VALID_MODES}, "
                f"got '{rectification_mode}'")
        self.mode = rectification_mode

        if distance_mode not in VALID_DISTANCE_MODES:
            raise ValueError(
                f"distance_mode must be one of {VALID_DISTANCE_MODES}, "
                f"got '{distance_mode}'")
        self.distance_mode = distance_mode

        # Track sub-component params explicitly so update_* methods can
        # rebuild from known state without having to introspect OpenCV.
        self.median_filter_size = self._sanitize_median(median_filter_size)

        # CLAHE state
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_grid_size = max(2, int(clahe_grid_size))
        self.clahe = None
        self._build_clahe()

        # WLS state
        self.wls_conf_threshold = int(wls_conf_threshold)
        self.wls_lambda = float(wls_lambda)
        self.wls_sigma = float(wls_sigma)
        self._wls_requested = bool(use_wls_filter)
        self.use_wls = self._wls_requested and _HAS_XIMGPROC
        self.wls_requested_but_unavailable = (
            self._wls_requested and not _HAS_XIMGPROC)

        # Rectification state
        self.H1 = None
        self.H2 = None
        self.F = None
        self.map1_l = self.map2_l = None
        self.map1_r = self.map2_r = None
        self.calibrated = (self.mode == "none")

        # Camera intrinsics
        self.K1 = None
        self.K2 = None
        self.D1 = None
        self.D2 = None
        self.R1_rect = None
        self.R2_rect = None
        self.P1 = None
        self.P2 = None
        self.image_size = None
        self.focal_length = None

        self._radial_scale = None
        self._radial_scale_for_size = None

        # SGBM + right matcher + WLS
        mode_key = str(sgbm_mode).lower().strip()
        if mode_key not in _SGBM_MODES:
            raise ValueError(
                f"sgbm_mode must be one of {VALID_SGBM_MODES}, "
                f"got '{sgbm_mode}'")
        self.sgbm_mode_name = mode_key

        self.sgbm_num_disparities = _multiple_of_16(sgbm_num_disparities)
        self.sgbm_block_size = _odd_clamp(sgbm_block_size, 3, 31)
        self.sgbm_disp12_max_diff = int(sgbm_disp12_max_diff)
        self.sgbm_uniqueness_ratio = int(sgbm_uniqueness_ratio)

        self.sgbm = None
        self.right_matcher = None
        self.wls = None
        self._build_sgbm()
        self._build_wls()

    # ------------------------------------------------------------------ #
    #  Builders                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize_median(size):
        mfs = int(size)
        if mfs <= 1:
            return 0
        if mfs % 2 == 0:
            mfs += 1
        # cv2.medianBlur on CV_32F only accepts ksize in {3, 5}.
        return min(mfs, 5)

    def _build_clahe(self):
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_grid_size, self.clahe_grid_size),
            )
        else:
            self.clahe = None

    def _build_sgbm(self):
        bs = self.sgbm_block_size
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.sgbm_num_disparities,
            blockSize=bs,
            P1=8 * 3 * bs * bs,
            P2=32 * 3 * bs * bs,
            disp12MaxDiff=self.sgbm_disp12_max_diff,
            uniquenessRatio=self.sgbm_uniqueness_ratio,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=_SGBM_MODES[self.sgbm_mode_name],
        )

    def _build_wls(self):
        """Rebuild right matcher + WLS filter to match the current SGBM.

        Must be called whenever SGBM is rebuilt, because the right
        matcher and WLS cache the matcher parameters at creation.
        """
        if self.use_wls:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.sgbm)
            self.wls = cv2.ximgproc.createDisparityWLSFilter(
                matcher_left=self.sgbm)
            self.wls.setLambda(self.wls_lambda)
            self.wls.setSigmaColor(self.wls_sigma)
        else:
            self.right_matcher = None
            self.wls = None

    # ------------------------------------------------------------------ #
    #  Runtime parameter updates (called from the node's param callback) #
    # ------------------------------------------------------------------ #
    def update_sgbm(self, num_disparities=None, block_size=None,
                    sgbm_mode=None, disp12_max_diff=None,
                    uniqueness_ratio=None):
        """Rebuild StereoSGBM (and the WLS pieces that depend on it).

        Any argument left as None keeps its current value.
        """
        if num_disparities is not None:
            self.sgbm_num_disparities = _multiple_of_16(num_disparities)
        if block_size is not None:
            self.sgbm_block_size = _odd_clamp(block_size, 3, 31)
        if sgbm_mode is not None:
            m = str(sgbm_mode).lower().strip()
            if m not in _SGBM_MODES:
                raise ValueError(
                    f"sgbm_mode must be one of {VALID_SGBM_MODES}")
            self.sgbm_mode_name = m
        if disp12_max_diff is not None:
            self.sgbm_disp12_max_diff = int(disp12_max_diff)
        if uniqueness_ratio is not None:
            self.sgbm_uniqueness_ratio = int(uniqueness_ratio)

        self._build_sgbm()
        # WLS holds a reference to the matcher's parameters at creation,
        # so it must be rebuilt when SGBM changes.
        self._build_wls()

    def update_wls(self, use_wls=None, lambda_val=None, sigma=None,
                   conf_threshold=None):
        """Tweak WLS parameters at runtime.

        - lambda/sigma updates poke the existing filter (cheap, no rebuild).
        - toggling use_wls rebuilds the filter object.
        """
        rebuild = False
        if use_wls is not None:
            new_req = bool(use_wls)
            new_use = new_req and _HAS_XIMGPROC
            if new_use != self.use_wls:
                self._wls_requested = new_req
                self.use_wls = new_use
                self.wls_requested_but_unavailable = (
                    new_req and not _HAS_XIMGPROC)
                rebuild = True
        if lambda_val is not None:
            self.wls_lambda = float(lambda_val)
            if self.wls is not None:
                self.wls.setLambda(self.wls_lambda)
        if sigma is not None:
            self.wls_sigma = float(sigma)
            if self.wls is not None:
                self.wls.setSigmaColor(self.wls_sigma)
        if conf_threshold is not None:
            self.wls_conf_threshold = int(conf_threshold)
        if rebuild:
            self._build_wls()

    def update_clahe(self, use_clahe=None, clip_limit=None, grid_size=None):
        if use_clahe is not None:
            self.use_clahe = bool(use_clahe)
        if clip_limit is not None:
            self.clahe_clip_limit = float(clip_limit)
        if grid_size is not None:
            self.clahe_grid_size = max(2, int(grid_size))
        self._build_clahe()

    def update_distance_mode(self, distance_mode):
        d = str(distance_mode).lower()
        if d not in VALID_DISTANCE_MODES:
            raise ValueError(
                f"distance_mode must be one of {VALID_DISTANCE_MODES}")
        self.distance_mode = d

    def update_median_filter_size(self, size):
        self.median_filter_size = self._sanitize_median(size)

    def update_baseline(self, baseline_mm):
        self.baseline_mm = float(baseline_mm)

    # ------------------------------------------------------------------ #
    #  Read-only flags                                                   #
    # ------------------------------------------------------------------ #
    @property
    def wls_enabled(self):
        return self.use_wls and self.wls is not None

    # ------------------------------------------------------------------ #
    #  Camera info                                                       #
    # ------------------------------------------------------------------ #
    def set_camera_info(self, K1, K2,
                        D1=None, D2=None,
                        R1=None, R2=None,
                        P1=None, P2=None,
                        image_size=None):
        self.K1 = np.asarray(K1, dtype=np.float64).reshape(3, 3)
        self.K2 = np.asarray(K2, dtype=np.float64).reshape(3, 3)
        if D1 is not None:
            self.D1 = np.asarray(D1, dtype=np.float64).ravel()
        if D2 is not None:
            self.D2 = np.asarray(D2, dtype=np.float64).ravel()
        if R1 is not None:
            self.R1_rect = np.asarray(R1, dtype=np.float64).reshape(3, 3)
        if R2 is not None:
            self.R2_rect = np.asarray(R2, dtype=np.float64).reshape(3, 3)
        if P1 is not None:
            self.P1 = np.asarray(P1, dtype=np.float64).reshape(3, 4)
        if P2 is not None:
            self.P2 = np.asarray(P2, dtype=np.float64).reshape(3, 4)
        if image_size is not None:
            self.image_size = (int(image_size[0]), int(image_size[1]))

        if self.mode == "calibrated" and self.P1 is not None and self.P2 is not None:
            self.focal_length = 0.5 * (self.P1[0, 0] + self.P2[0, 0])
        else:
            self.focal_length = 0.5 * (self.K1[0, 0] + self.K2[0, 0])

        if self.mode == "calibrated":
            self._build_calibrated_maps()

        self._radial_scale = None
        self._radial_scale_for_size = None

    # ------------------------------------------------------------------ #
    #  Calibration                                                       #
    # ------------------------------------------------------------------ #
    def _build_calibrated_maps(self):
        if (self.K1 is None or self.K2 is None or
                self.R1_rect is None or self.R2_rect is None or
                self.P1 is None or self.P2 is None or
                self.image_size is None):
            return False
        D1 = self.D1 if self.D1 is not None else np.zeros(5)
        D2 = self.D2 if self.D2 is not None else np.zeros(5)
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.K1, D1, self.R1_rect, self.P1,
            self.image_size, cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.K2, D2, self.R2_rect, self.P2,
            self.image_size, cv2.CV_32FC1)
        self.calibrated = True
        return True

    def calibrate(self, img_left, img_right):
        if self.mode == "none":
            self.calibrated = True
            return True
        if self.mode == "calibrated":
            return self._build_calibrated_maps()

        if img_left is None or img_right is None:
            return False
        if img_left.shape[:2] != img_right.shape[:2]:
            return False

        gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) \
            if img_left.ndim == 3 else img_left
        gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) \
            if img_right.ndim == 3 else img_right

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_l, None)
        kp2, des2 = sift.detectAndCompute(gray_r, None)
        if des1 is None or des2 is None:
            return False
        if len(kp1) < 8 or len(kp2) < 8:
            return False

        bf = cv2.BFMatcher()
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        chosen = matches[: self.max_features]
        if len(chosen) < 8:
            return False

        pairs = sift_features_to_array(chosen, kp1, kp2)

        F_best, inliers = get_inliers(pairs)
        if F_best is None or inliers.shape[0] < 8:
            return False

        set1 = inliers[:, 0:2].astype(np.float32)
        set2 = inliers[:, 2:4].astype(np.float32)

        h, w = gray_l.shape[:2]
        ok, H1, H2 = cv2.stereoRectifyUncalibrated(
            set1.reshape(-1, 1, 2),
            set2.reshape(-1, 1, 2),
            F_best,
            imgSize=(w, h),
        )
        if not ok:
            return False

        self.F = F_best
        self.H1 = H1
        self.H2 = H2
        self.calibrated = True
        return True

    def reset_calibration(self):
        if self.mode == "none":
            return
        self.calibrated = False
        self.H1 = self.H2 = self.F = None
        self.map1_l = self.map2_l = None
        self.map1_r = self.map2_r = None

    # ------------------------------------------------------------------ #
    #  Per-frame pipeline                                                #
    # ------------------------------------------------------------------ #
    def rectify(self, img_left, img_right):
        if self.mode == "none":
            return img_left, img_right
        if not self.calibrated:
            return None, None
        if self.mode == "uncalibrated":
            h, w = img_left.shape[:2]
            rect_l = cv2.warpPerspective(img_left, self.H1, (w, h))
            rect_r = cv2.warpPerspective(img_right, self.H2, (w, h))
            return rect_l, rect_r
        rect_l = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return rect_l, rect_r

    def _to_gray(self, img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def compute_disparity(self, rect_left, rect_right):
        g_l = self._to_gray(rect_left)
        g_r = self._to_gray(rect_right)

        if self.use_clahe and self.clahe is not None:
            g_l = self.clahe.apply(g_l)
            g_r = self.clahe.apply(g_r)

        disp_left_16 = self.sgbm.compute(g_l, g_r)

        if self.use_wls and self.wls is not None:
            disp_right_16 = self.right_matcher.compute(g_r, g_l)
            filtered_16 = self.wls.filter(
                disp_left_16, g_l, disparity_map_right=disp_right_16)
            disp = filtered_16.astype(np.float32) / 16.0

            if self.wls_conf_threshold > 0:
                try:
                    conf = self.wls.getConfidenceMap()
                    if conf is not None and conf.shape == disp.shape:
                        disp[conf < float(self.wls_conf_threshold)] = -1.0
                except cv2.error:
                    pass
        else:
            disp = disp_left_16.astype(np.float32) / 16.0

        if self.median_filter_size >= 3:
            disp = cv2.medianBlur(disp, self.median_filter_size)

        return disp

    # ------------------------------------------------------------------ #
    #  Depth / distance                                                  #
    # ------------------------------------------------------------------ #
    def _ensure_radial_scale(self, shape):
        h, w = shape[:2]
        if (self._radial_scale is not None and
                self._radial_scale_for_size == (w, h)):
            return self._radial_scale

        if self.mode == "calibrated" and self.P1 is not None:
            fx = float(self.P1[0, 0])
            fy = float(self.P1[1, 1])
            cx = float(self.P1[0, 2])
            cy = float(self.P1[1, 2])
        elif self.K1 is not None:
            fx = float(self.K1[0, 0])
            fy = float(self.K1[1, 1])
            cx = float(self.K1[0, 2])
            cy = float(self.K1[1, 2])
        else:
            fx = fy = max(w, h)
            cx = (w - 1) * 0.5
            cy = (h - 1) * 0.5

        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        x_n = (uu - cx) / fx
        y_n = (vv - cy) / fy
        scale = np.sqrt(1.0 + x_n * x_n + y_n * y_n).astype(np.float32)

        self._radial_scale = scale
        self._radial_scale_for_size = (w, h)
        return scale

    def compute_depth_mm(self, disparity):
        if self.focal_length is None:
            return None

        Z = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0.5
        Z[valid] = (self.baseline_mm * self.focal_length) / disparity[valid]

        if self.distance_mode == "z":
            return Z

        scale = self._ensure_radial_scale(disparity.shape)
        R = Z * scale
        return R

    @staticmethod
    def region_median_depth(depth_map, mask):
        if depth_map is None or mask is None:
            return None
        sel = depth_map[(mask > 0) & (depth_map > 0)]
        if sel.size == 0:
            return None
        return float(np.median(sel))