"""
=============================================================================
 vision_main_2.py  -  [진단/테스트 전용]  22mL_cylinder 미검출 원인 분석
=============================================================================
 * vision_main.py 의 detect_circle_by_diff 로직을 "그대로" 유지합니다.
   (score_map / threshold / 최대 blob 1개 / distanceTransform / depth 단차)
   → 이미 잘 되는 제품(1L_marineli 등)의 동작은 100% 동일합니다.

 * 딱 한 곳만 바뀝니다: 반지름 범위(MIN/MAX_RADIUS_PX)를
   product.json 의 per-product override 로 읽을 수 있게 했습니다.
   override 가 없으면 기존 공식(300*mm/150, 800*mm/150)을 그대로 씁니다.

 * 실행하면 각 단계별로 왜 통과/탈락했는지 전부 출력하고,
   디버그 이미지(_debug_*.png)를 남깁니다.
=============================================================================
"""

import cv2
cv2.setNumThreads(0)
import numpy as np
import json
import math
import time
import os
import sys
from pathlib import Path

try:
    import zxingcpp
    HAS_ZXING = True
except Exception:
    HAS_ZXING = False


# =============================================================================
# 진단 로그 헬퍼
# =============================================================================
# QR 이 안 읽히는 환경에서 fix2 를 테스트할 때만 쓰는 수동 QR 중심 좌표.
# 실기에서는 None 으로 두면 됨 (_read_qr_robust 결과를 그대로 사용).
QR_HINT = None


def log(msg):    print(f"    {msg}")
def ok(msg):     print(f"    [OK]   {msg}")
def ng(msg):     print(f"    [DROP] {msg}")
def hdr(msg):    print(f"\n--- {msg} " + "-" * max(0, 60 - len(msg)))


class VisionCoreTest:
    def __init__(self, config_path=None, product_path=None):
        curr_dir = Path(__file__).resolve().parent
        root = curr_dir.parent if curr_dir.name == "VISION" else curr_dir

        self.vision_config_file = Path(config_path) if config_path else root / "DB" / "vision" / "vision_config.json"
        self.product_config_file = Path(product_path) if product_path else root / "DB" / "product" / "product.json"

        self.v_config = self.load_vision_config()
        self.base_width = 3840.0
        self.debug_dir = curr_dir / "_debug"
        self.debug_dir.mkdir(exist_ok=True)

    def load_vision_config(self):
        if not self.vision_config_file.exists():
            print(f"[WARN] vision_config.json 없음 -> 기본값 사용: {self.vision_config_file}")
            return {"MIN_RADIUS_PX": 200, "MAX_RADIUS_PX": 800, "COLOR_WEIGHT_THRESHOLD": 20,
                    "QR_SCALE_FACTOR": 1.0, "Z_OFFSET_METERS": -0.04}
        with open(self.vision_config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.setdefault("COLOR_WEIGHT_THRESHOLD", 20)
        return data

    # =========================================================================
    # ★ 변경점 (딱 여기 한 곳)
    #   product.json 의 beaker[name] 에 min_radius_px / max_radius_px 가 있으면
    #   그 값을 쓰고, 없으면 기존 vision_main.py 공식을 그대로 사용.
    #   -> 기존 제품은 override 키가 없으므로 동작이 전혀 바뀌지 않음.
    # =========================================================================
    def apply_product_config(self, beaker_name, is_calib=False):
        target_color = "white"
        if not beaker_name:
            return target_color

        if not self.product_config_file.exists():
            print(f"[WARN] product.json 없음: {self.product_config_file}")
            return target_color

        with open(self.product_config_file, 'r', encoding='utf-8') as f:
            p_data = json.load(f)

        if is_calib:
            b_data = p_data.get("calibration", {}).get(beaker_name, {})
            target_mm = b_data.get("anchor_mm", b_data.get("target_mm", 150.0))
        else:
            b_data = p_data.get("beaker", {}).get(beaker_name, {})
            target_mm = b_data.get("target_mm", 150.0)

        target_color = b_data.get("color", "white")

        # --- 기존 공식 (원본 유지) ---
        if target_mm > 0:
            ratio = target_mm / 150.0
            self.v_config["MIN_RADIUS_PX"] = int(300 * ratio)
            self.v_config["MAX_RADIUS_PX"] = int(800 * ratio)

        # --- per-product override (있을 때만) ---
        ov_min = b_data.get("min_radius_px")
        ov_max = b_data.get("max_radius_px")
        if ov_min: self.v_config["MIN_RADIUS_PX"] = int(ov_min)
        if ov_max: self.v_config["MAX_RADIUS_PX"] = int(ov_max)

        src = "product.json override" if (ov_min or ov_max) else "기존 공식(300/800 * mm/150)"
        log(f"제품='{beaker_name}'  color={target_color}  target_mm={target_mm}")
        log(f"반지름 범위(원본해상도) = {self.v_config['MIN_RADIUS_PX']} ~ {self.v_config['MAX_RADIUS_PX']} px   [{src}]")
        return target_color

    # =========================================================================
    # vision_main.py 의 detect_circle_by_diff 와 로직 100% 동일 + 단계별 로그
    # =========================================================================
    def _select_blob(self, contours, h, w, scale, min_r, max_r, qr_xy):
        """
        fix2 전용 blob 선택기.

        '가장 큰 blob' 은 조명이 바뀌면 지그 반사광에 밀린다. 대신 3단계로 거른다.
          1) 반지름   : 제품 규격 범위 밖이면 탈락
          2) 원형도   : 뚜껑은 0.86~0.87, 반사광은 0.41~0.50 -> 깨끗하게 갈림
          3) 최종선택 : QR 을 품은 blob 우선, 없으면 '화면 중앙에 가장 가까운' blob
                        (로봇이 지그 위에 카메라를 정렬하므로 뚜껑은 항상 중앙 근처.
                         실측: 뚜껑 중심거리 31~50px vs 반사광 278~2203px)
        면적은 더 이상 선택 기준이 아니다.
        """
        min_full = self.v_config["MIN_RADIUS_PX"]
        max_full = self.v_config["MAX_RADIUS_PX"]
        r_lo, r_hi = min_full * 0.6, max_full * 1.5
        circ_min = self.v_config.get("WHITE_CIRCULARITY_MIN", 0.65)

        img_cx, img_cy = w / 2.0, h / 2.0
        qx = qy = None
        if qr_xy is not None:
            qx, qy = qr_xy[0] * scale, qr_xy[1] * scale

        cands = []
        for cnt in contours:
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [cnt], -1, 255, -1)
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            _, inner, _, loc = cv2.minMaxLoc(dt)
            _, _, ww, hh = cv2.boundingRect(cnt)
            r_full = int(((inner + max(ww, hh) / 2.0) / 2.0) / scale)
            cxf, cyf = int(loc[0] / scale), int(loc[1] / scale)

            if not (r_lo <= r_full <= r_hi):
                log(f"  탈락 r={r_full}px @({cxf},{cyf}) : 반지름 밖 (허용 {int(r_lo)}~{int(r_hi)})")
                continue

            area = cv2.contourArea(cnt)
            per = cv2.arcLength(cnt, True)
            circ = (4 * math.pi * area) / (per * per) if per > 0 else 0.0
            if circ < circ_min:
                log(f"  탈락 r={r_full}px @({cxf},{cyf}) : 원형도 {circ:.2f} < {circ_min}")
                continue

            d_ctr = math.hypot(loc[0] - img_cx, loc[1] - img_cy)
            has_qr, d_qr = False, 1e9
            if qx is not None:
                d_qr = math.hypot(loc[0] - qx, loc[1] - qy)
                has_qr = (cv2.pointPolygonTest(cnt, (float(qx), float(qy)), False) >= 0
                          or d_qr <= max(inner, 10))

            log(f"  후보 r={r_full}px @({cxf},{cyf}) 원형도={circ:.2f} "
                f"중심거리={d_ctr/scale:.0f}px QR포함={has_qr}")
            cands.append({"cnt": cnt, "r": r_full, "circ": circ,
                          "d_ctr": d_ctr, "d_qr": d_qr, "has_qr": has_qr})

        if not cands:
            return None

        qr_hits = [c for c in cands if c["has_qr"]]
        if qr_hits:
            qr_hits.sort(key=lambda c: c["d_qr"])
            best = qr_hits[0]
            ok(f"QR 앵커로 선택: r={best['r']}px (QR 거리 {best['d_qr']/scale:.0f}px)")
        else:
            cands.sort(key=lambda c: c["d_ctr"])
            best = cands[0]
            why = "QR 없음" if qx is None else "QR 포함 blob 없음"
            ok(f"{why} -> 화면중앙 최근접으로 선택: r={best['r']}px "
               f"(중심거리 {best['d_ctr']/scale:.0f}px)")
        return best["cnt"]

    def detect_circle_by_diff(self, color_img, depth_img, target_color="white",
                              dump=False, tag="", mode="orig", qr_xy=None):
        """
        mode="orig" : vision_main.py 와 100% 동일
        mode="fix"  : white 일 때만 아래 2가지 추가 (다른 color 는 orig 와 동일)
                      (a) 밝기 게이트  min(R,G,B) > WHITE_MIN_BRIGHTNESS
                          -> 파란 랩/비닐 처럼 'B-R 은 크지만 어두운' 것 제거
                      (b) MORPH_CLOSE -> 뚜껑 위 QR 스티커가 뚫어놓은 구멍 메움
        """
        self.last_binary = None
        self.last_score = None
        self.last_info = {}
        scale = 0.5
        small_color = cv2.resize(color_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = small_color.shape[:2]

        scale_ratio = w / (self.base_width * scale)
        min_r = max(10, int(self.v_config["MIN_RADIUS_PX"] * scale_ratio * scale))
        max_r = max(20, int(self.v_config["MAX_RADIUS_PX"] * scale_ratio * scale))
        color_threshold = self.v_config.get("COLOR_WEIGHT_THRESHOLD", 20)

        log(f"축소영상 {w}x{h} | min_r={min_r} max_r={max_r} (축소기준) | thr={color_threshold}")

        # 1. score map (원본 그대로)
        B, G, R = cv2.split(small_color.astype(np.float32))
        if   target_color == "white":  score_map = B - R
        elif target_color == "black":  score_map = 255.0 - ((B + G + R) / 3.0)
        elif target_color == "blue":   score_map = B - ((G + R) / 2.0)
        elif target_color == "red":    score_map = R - ((B + G) / 2.0)
        elif target_color == "purple": score_map = ((B + R) / 2.0) - G
        elif target_color == "cyan":   score_map = ((B + G) / 2.0) - R
        else:                          score_map = (B + G + R) / 3.0

        # 2. 이진화
        score_u8 = np.clip(score_map, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(score_u8, color_threshold, 255, cv2.THRESH_BINARY)

        # ---- mode="fix" 추가분 (white 전용) ----------------------------------
        if mode in ("fix", "fix2") and target_color == "white":
            bright_thr = self.v_config.get("WHITE_MIN_BRIGHTNESS", 90)
            min_rgb = np.minimum(np.minimum(R, G), B).astype(np.uint8)
            _, bright_mask = cv2.threshold(min_rgb, bright_thr, 255, cv2.THRESH_BINARY)
            binary = cv2.bitwise_and(binary, bright_mask)
            log(f"[fix] 밝기 게이트 적용: min(R,G,B) > {bright_thr}")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        if mode in ("fix", "fix2") and target_color == "white":
            ck = self.v_config.get("WHITE_CLOSE_KSIZE", 25)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
            log(f"[fix] MORPH_CLOSE({ck}) 적용: QR 스티커 구멍 메움")
        # ---------------------------------------------------------------------

        self.last_score = score_u8
        self.last_binary = binary
        self.last_info = {"min_r": min_r, "max_r": max_r, "thr": color_threshold,
                          "scale": scale, "mode": mode}

        if dump:
            cv2.imwrite(str(self.debug_dir / f"01_score{tag}.png"), score_u8)
            cv2.imwrite(str(self.debug_dir / f"02_binary{tag}.png"), binary)

        # 3. contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            ng("contour 0개 -> 색상 threshold 가 너무 높음")
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        log(f"contour {len(contours)}개, 상위 3개 면적: " +
            ", ".join(f"{cv2.contourArea(c):.0f}" for c in contours[:3]))

        # =====================================================================
        # ★ mode="fix2" : blob 선택 방식만 교체 (뒷단 계산은 전부 동일)
        #   문제 - 사이트마다 조명이 달라서 지그 반사광이 색상 게이트를 통과함
        #          Area1 지그 B-R=+9  (thr 20 미만 -> 통과 못함)
        #          Area2 지그 B-R=+40, minRGB=144  -> 두 게이트 다 통과!
        #   색상만으로는 절대 못 거름. 그래서 아래 2개를 추가로 쓴다.
        #     (1) 최종 반지름이 제품 규격 범위 안에 있는 것만 후보로
        #     (2) QR 은 항상 뚜껑 위에 있으므로, QR 중심을 품은 blob 을 최우선
        #   QR 이 없으면 기존과 동일하게 '가장 큰 blob' 으로 폴백.
        # =====================================================================
        if mode == "fix2":
            largest_cnt = self._select_blob(contours, h, w, scale, min_r, max_r, qr_xy)
            if largest_cnt is None:
                ng("fix2: 조건을 만족하는 후보 blob 없음")
                return None
        else:
            largest_cnt = contours[0]

        area = cv2.contourArea(largest_cnt)
        area_limit = math.pi * (min_r ** 2) * 0.2
        if area < area_limit:
            ng(f"최대 blob 면적 {area:.0f} < 기준 {area_limit:.0f}")
            return None
        ok(f"최대 blob 면적 {area:.0f} >= {area_limit:.0f}")

        # distanceTransform
        mask_for_dt = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_for_dt, [largest_cnt], -1, 255, -1)
        dist_transform = cv2.distanceTransform(mask_for_dt, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        cx, cy = max_loc
        inner_radius = max_val

        lo, hi = min_r * 0.5, max_r * 1.5
        margin = (hi - inner_radius) / hi * 100 if hi else 0
        self.last_info.update({"inner": inner_radius, "lo": lo, "hi": hi, "margin": margin,
                               "cand_cx": int(cx / scale), "cand_cy": int(cy / scale)})
        log(f"inner_radius={inner_radius:.1f} / 허용 {lo:.1f}~{hi:.1f} (상한여유 {margin:.1f}%)")
        if not (lo <= inner_radius <= hi):
            ng(f"★ 반지름 범위 탈락! inner_radius={inner_radius:.1f}, 허용 {lo:.1f}~{hi:.1f}")
            ng(f"  -> product.json 의 min_radius_px/max_radius_px 로 보정 필요. "
               f"권장 max_radius_px >= {int(inner_radius / scale_ratio / scale * 1.4)}")
            return None
        if margin < 25:
            log(f"[!] 상한 여유 {margin:.1f}% 밖에 안 됨 -> 조명/거리 조금만 변해도 실패함")
        ok("반지름 범위 통과")

        _, _, w_w, w_h = cv2.boundingRect(largest_cnt)
        outer_radius = max(w_w, w_h) / 2.0
        final_radius = (inner_radius + outer_radius) / 2.0

        dx, dy, dr = int(cx / scale), int(cy / scale), int(final_radius / scale)
        log(f"원본좌표 center=({dx},{dy})  r={dr}px  (inner={inner_radius:.0f} outer={outer_radius:.0f})")

        if dump:
            vis = color_img.copy()
            cv2.circle(vis, (dx, dy), dr, (0, 255, 0), 6)
            cv2.drawMarker(vis, (dx, dy), (0, 0, 255), cv2.MARKER_CROSS, 80, 6)
            cv2.imwrite(str(self.debug_dir / f"03_detect{tag}.png"), cv2.resize(vis, (1280, 720)))

        # 4. depth 단차 검사 (원본 그대로)
        if depth_img is None:
            log("[SKIP] depth 없음(이미지 파일 테스트) -> 단차 검사 생략")
            return (dx, dy, dr)

        if depth_img.shape[:2] != color_img.shape[:2]:
            ng(f"★ depth({depth_img.shape[:2]}) 와 color({color_img.shape[:2]}) 해상도 불일치! "
               f"HW align 확인 필요 (좌표계가 어긋나면 단차검사가 항상 실패함)")

        roi_mask_in = np.zeros(depth_img.shape, dtype=np.uint8)
        cv2.circle(roi_mask_in, (dx, dy), int(dr * 0.7), 255, -1)
        roi_mask_out = np.zeros(depth_img.shape, dtype=np.uint8)
        cv2.circle(roi_mask_out, (dx, dy), int(dr * 1.3), 255, -1)
        cv2.circle(roi_mask_out, (dx, dy), int(dr * 1.05), 0, -1)

        z_in_vals = depth_img[(roi_mask_in == 255) & (depth_img > 0)]
        z_out_vals = depth_img[(roi_mask_out == 255) & (depth_img > 0)]
        log(f"depth 유효픽셀 in={len(z_in_vals)} out={len(z_out_vals)} (각각 >50 필요)")

        if not (len(z_in_vals) > 50 and len(z_out_vals) > 50):
            ng("★ depth 유효픽셀 부족 -> 22mL 처럼 원이 작으면 여기서 자주 탈락함")
            ng(f"  -> in ring r={int(dr*0.7)}px, out ring r={int(dr*1.05)}~{int(dr*1.3)}px")
            return None

        z_in = float(np.median(z_in_vals))
        z_out = float(np.median(z_out_vals))
        depth_diff = abs(z_out - z_in)
        log(f"z_in={z_in*1000:.1f}mm z_out={z_out*1000:.1f}mm 단차={depth_diff*1000:.1f}mm (허용 5~450mm)")
        if not (0.005 <= depth_diff <= 0.45):
            ng("★ 단차 검사 탈락 -> 22mL 뚜껑은 단차가 5mm 미만일 수 있음")
            return None
        ok("단차 검사 통과")
        return (dx, dy, dr)

    # =========================================================================
    # QR (vision_main.py 원본 그대로)
    # =========================================================================
    def _read_qr_robust(self, color_img):
        if not HAS_ZXING:
            log("[SKIP] zxingcpp 미설치")
            return None
        fmt = zxingcpp.BarcodeFormat.QRCode | zxingcpp.BarcodeFormat.DataMatrix | zxingcpp.BarcodeFormat.MicroQRCode
        h, w = color_img.shape[:2]
        roi_w, roi_h = int(w * 0.40), int(h * 0.50)
        sx, sy = (w - roi_w) // 2, (h - roi_h) // 2
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.rectangle(mask, (sx, sy), (sx + roi_w, sy + roi_h), 255, -1)
        safe_gray = cv2.bitwise_and(gray, mask)

        stages = []
        stages.append(("1.gray", safe_gray))
        inv = 1.0 / 0.5
        table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
        stages.append(("2.gamma", cv2.LUT(safe_gray, table)))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(safe_gray)
        stages.append(("3.clahe", enhanced))
        at = cv2.bitwise_and(cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 41, 5), mask)
        stages.append(("4.adaptive", at))
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, bn = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bn = cv2.bitwise_and(bn, mask)
        stages.append(("5.otsu", bn))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        stages.append(("6.close", cv2.bitwise_and(cv2.morphologyEx(bn, cv2.MORPH_CLOSE, k), mask)))

        for name, im in stages:
            b = zxingcpp.read_barcodes(im, formats=fmt)
            if b:
                ok(f"QR 성공 @ {name} -> '{b[0].text}'")
                return b
            log(f"{name} 실패")
        ng("QR 전 단계 실패")
        return None


# =============================================================================
# 결과 화면 표시
# =============================================================================
def build_view(color_img, res_orig, res_fix, res_fix2, qr_pts, qr_xy, qr_text,
               beaker_name, target_color):
    """ORIG=빨강, FIX=주황, FIX2=초록 을 한 화면에 겹쳐 그림"""
    vis = color_img.copy()
    h, w = vis.shape[:2]
    th = max(3, int(w / 800))
    fs = max(1.0, 1.8 * (w / 3840.0))

    (c_o, i_o), (c_f, i_f), (c_f2, i_f2) = res_orig, res_fix, res_fix2

    # 화면 중심 십자 (노랑)
    ccx, ccy = w // 2, h // 2
    cv2.line(vis, (ccx - 80, ccy), (ccx + 80, ccy), (0, 255, 255), th)
    cv2.line(vis, (ccx, ccy - 80), (ccx, ccy + 80), (0, 255, 255), th)

    # 허용 반지름 범위 (파랑=MIN, 주황=MAX) - 화면 중앙 기준
    sc = i_f2.get("scale", 0.5)
    min_full = int(i_f2.get("lo", i_o.get("lo", 0)) / sc)
    max_full = int(i_f2.get("hi", i_o.get("hi", 0)) / sc)
    if max_full > 0:
        cv2.circle(vis, (ccx, ccy), min_full, (255, 128, 0), th)
        cv2.circle(vis, (ccx, ccy), max_full, (0, 128, 255), th)

    if qr_pts is not None:
        cv2.polylines(vis, [qr_pts], True, (255, 0, 255), th)
    if qr_xy is not None:
        cv2.drawMarker(vis, tuple(qr_xy), (255, 0, 255), cv2.MARKER_CROSS, 120, th)

    if c_o:
        cv2.circle(vis, (c_o[0], c_o[1]), c_o[2], (0, 0, 255), th)          # 빨강
    if c_f:
        cv2.circle(vis, (c_f[0], c_f[1]), c_f[2], (0, 165, 255), th + 1)    # 주황
    if c_f2:
        cv2.circle(vis, (c_f2[0], c_f2[1]), c_f2[2], (0, 255, 0), th + 3)   # 초록
        cv2.line(vis, (c_f2[0] - 60, c_f2[1]), (c_f2[0] + 60, c_f2[1]), (0, 255, 0), th)
        cv2.line(vis, (c_f2[0], c_f2[1] - 60), (c_f2[0], c_f2[1] + 60), (0, 255, 0), th)

    lines = [
        (f"product: {beaker_name} ({target_color})   allow R: {min_full}~{max_full}px", (255, 255, 255)),
        ("[ORIG] " + (f"X:{c_o[0]} Y:{c_o[1]} R:{c_o[2]}px" if c_o else "FAIL"), (80, 80, 255)),
        ("[FIX ] " + (f"X:{c_f[0]} Y:{c_f[1]} R:{c_f[2]}px" if c_f else "FAIL"), (0, 165, 255)),
        ("[FIX2] " + (f"X:{c_f2[0]} Y:{c_f2[1]} R:{c_f2[2]}px" if c_f2 else "FAIL"), (0, 255, 0)),
    ]
    if c_f2:
        lines.append((f"        offset from center: {int(math.hypot(c_f2[0]-ccx, c_f2[1]-ccy))}px", (0, 255, 0)))
    lines.append((f"QR: {qr_text}", (255, 0, 255)))

    y = 90
    for txt, col in lines:
        cv2.putText(vis, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 4)
        cv2.putText(vis, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)
        y += int(fs * 55)
    return vis


def show_windows(vis, bin_orig, bin_fix):
    out = Path(__file__).resolve().parent / "_debug" / "result_view.png"
    cv2.imwrite(str(out), vis)
    print(f"\n  결과 이미지 저장: {out}")
    try:
        _show_windows(vis, bin_orig, bin_fix)
    except cv2.error as e:
        print(f"  [INFO] GUI 사용 불가(headless) - 저장 파일로 확인: {e}")


def _show_windows(vis, bin_orig, bin_fix):
    cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RESULT", 1280, 720)
    cv2.imshow("RESULT", vis)

    if bin_orig is not None and bin_fix is not None:
        panel = np.hstack([cv2.resize(bin_orig, (640, 360)),
                           cv2.resize(bin_fix, (640, 360))])
        cv2.putText(panel, "BINARY - ORIG", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
        cv2.putText(panel, "BINARY - FIX", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
        cv2.namedWindow("BINARY  ORIG vs FIX", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BINARY  ORIG vs FIX", 1280, 360)
        cv2.imshow("BINARY  ORIG vs FIX", panel)

    print("\n  [창 조작]  아무 키나 = 종료 / s = 화면 저장")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            p = Path(__file__).resolve().parent / "_debug" / "result_view.png"
            cv2.imwrite(str(p), vis)
            print(f"  저장됨: {p}")
            continue
        break
    cv2.destroyAllWindows()


# =============================================================================
# 실행부
# =============================================================================
def run_test(image_path, beaker_name, sweep=True):
    print("=" * 70)
    print(f" 이미지: {Path(image_path).name}")
    print(f" 제품  : {beaker_name}")
    print("=" * 70)

    color_img = cv2.imread(image_path)
    if color_img is None:
        print(f"[ERROR] 이미지 읽기 실패: {image_path}")
        return

    v = VisionCoreTest()

    hdr("STEP 0. product.json 파라미터")
    target_color = v.apply_product_config(beaker_name)

    hdr("STEP 1. QR 검출")
    barcodes = v._read_qr_robust(color_img)
    qr_text, qr_pts = "FAIL", None
    if barcodes:
        qr_text = barcodes[0].text
        p = barcodes[0].position
        qr_pts = np.array([[q.x, q.y] for q in
                           [p.top_left, p.top_right, p.bottom_right, p.bottom_left]], dtype=np.int32)

    qr_xy = (int(np.mean(qr_pts[:, 0])), int(np.mean(qr_pts[:, 1]))) if qr_pts is not None else None
    if qr_xy is None and QR_HINT:
        qr_xy = QR_HINT
        log(f"[테스트] QR 미검출 -> 수동 지정 QR 중심 사용: {qr_xy}")

    hdr("STEP 2-A. [ORIG] vision_main.py 최초 로직 (색상 B-R + 최대 blob)")
    c_orig = v.detect_circle_by_diff(color_img, None, target_color, dump=True, tag="_orig", mode="orig")
    i_orig, bin_orig = v.last_info, v.last_binary
    print(f"  ==> ORIG: {'성공 ' + str(c_orig) if c_orig else '실패'}")

    hdr("STEP 2-B. [FIX] + 밝기 게이트 + CLOSE (현재 vision_main.py 적용본)")
    c_fix = v.detect_circle_by_diff(color_img, None, target_color, dump=True, tag="_fix", mode="fix")
    i_fix, bin_fix = v.last_info, v.last_binary
    print(f"  ==> FIX : {'성공 ' + str(c_fix) if c_fix else '실패'}")

    hdr("STEP 2-C. [FIX2] + 반지름 필터 + QR 앵커 선택")
    c_fix2 = v.detect_circle_by_diff(color_img, None, target_color, dump=True, tag="_fix2",
                                     mode="fix2", qr_xy=qr_xy)
    i_fix2, bin_fix2 = v.last_info, v.last_binary
    print(f"  ==> FIX2: {'성공 ' + str(c_fix2) if c_fix2 else '실패'}")

    vis = build_view(color_img, (c_orig, i_orig), (c_fix, i_fix), (c_fix2, i_fix2),
                     qr_pts, qr_xy, qr_text, beaker_name, target_color)

    if not sweep:
        show_windows(vis, bin_fix, bin_fix2)
        return

    hdr("STEP 3. 파라미터 스윕 (fix2 모드)")
    import io, contextlib
    base_b = v.v_config.get("WHITE_MIN_BRIGHTNESS", 90)
    base_c = v.v_config.get("WHITE_CLOSE_KSIZE", 25)
    print(f"  {'밝기게이트':>8} | {'CLOSE':>5} | 결과")
    print("  " + "-" * 50)
    for b in [0, 70, 90, 110]:
        for ck in [5, 15, 25, 35]:
            v.v_config["WHITE_MIN_BRIGHTNESS"] = b
            v.v_config["WHITE_CLOSE_KSIZE"] = ck
            with contextlib.redirect_stdout(io.StringIO()):
                c = v.detect_circle_by_diff(color_img, None, target_color, mode="fix2", qr_xy=qr_xy)
            print(f"  {b:>8} | {ck:>5} | {('OK  ' + str(c)) if c else 'FAIL'}")
    v.v_config["WHITE_MIN_BRIGHTNESS"] = base_b
    v.v_config["WHITE_CLOSE_KSIZE"] = base_c

    print(f"\n  디버그 이미지 저장 위치: {v.debug_dir}")

    hdr("STEP 4. 결과 화면 표시")
    show_windows(vis, bin_fix, bin_fix2)


if __name__ == "__main__":
    # 인자 없이 실행하면 VISION 폴더의 모든 jpg 를 순서대로 검사
    BEAKER = "22mL_cylinder"
    curr = Path(__file__).resolve()
    vdir = curr.parent
    root = vdir.parent if vdir.name == "VISION" else vdir

    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
        if len(sys.argv) > 2:
            BEAKER = sys.argv[2]
        if not files[0].exists():
            for c in [vdir / sys.argv[1], root / "Log" / "Image" / sys.argv[1], root / sys.argv[1]]:
                if c.exists():
                    files = [c]
                    break
    else:
        files = sorted(vdir.glob("*.jpg"))

    if not files or not files[0].exists():
        print("[ERROR] 검사할 이미지가 없습니다. VISION 폴더에 jpg 를 두세요.")
        sys.exit(1)

    for i, f in enumerate(files):
        run_test(str(f), BEAKER, sweep=(len(files) == 1))
        if i < len(files) - 1:
            print("\n\n")
