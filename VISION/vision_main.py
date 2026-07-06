import cv2
cv2.setNumThreads(0)
import zxingcpp
import numpy as np
import json
import math
import time
from pathlib import Path
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode, Context, OBPropertyID, OBLogLevel

class VisionCore:
    def __init__(self, config_path=None, product_path=None): 
        # 1. Vision Config 경로 강제 고정 (DB/vision/vision_config.json)
        if config_path is not None: 
            self.vision_config_file = Path(config_path)
        else:
            curr_dir = Path(__file__).resolve().parent
            if curr_dir.name == "VISION":
                self.vision_config_file = curr_dir.parent / "DB" / "vision" / "vision_config.json"
            else:
                self.vision_config_file = curr_dir / "DB" / "vision" / "vision_config.json"
        
        # 2. Product Config 경로 강제 고정 (DB/product/product.json)
        if product_path is not None:
            self.product_config_file = Path(product_path)
        else:
            curr_dir = Path(__file__).resolve().parent
            if curr_dir.name == "VISION":
                self.product_config_file = curr_dir.parent / "DB" / "product" / "product.json"
            else:
                self.product_config_file = curr_dir / "DB" / "product" / "product.json"
        
        self.v_config = self.load_vision_config()
        self.reset_ema()
        self.ALPHA = 0.15
        self.base_width = 3840.0
        
        self.ctx = Context()
        self.ctx.set_logger_level(OBLogLevel.ERROR)
        self.device = None
        self.on_image_ready = None
    def connect_camera(self):
        try:
            ip = self.v_config['CAMERA_IP']
            port = self.v_config['CAMERA_PORT']
            print(f"[VISION] 카메라 네트워크 연결 시도 중 ({ip})...")
            self.device = self.ctx.create_net_device(ip, port)
            self._apply_camera_settings(self.device)
            print("[VISION] 카메라 연결 성공.")
            return True
        except Exception as e:
            print(f"[VISION FATAL] 카메라 연결 실패: {e}")
            self.device = None
            return False

    def load_vision_config(self):
        self.vision_config_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.vision_config_file.exists():
            d = {
                "CAMERA_IP": "192.168.1.11", "CAMERA_PORT": 8090, "EDGE_SENSITIVITY": 80, "CIRCLE_THRESHOLD": 200,
                "MIN_DEPTH_DIFF_METERS": 0.015, "MIN_RADIUS_PX": 200, "MAX_RADIUS_PX": 800, "Z_OFFSET_METERS": -0.04,
                "COLOR_WIDTH": 3840, "COLOR_HEIGHT": 2160, "COLOR_FPS": 5, "DEPTH_WIDTH": 1024, "DEPTH_HEIGHT": 1024,
                "DEPTH_FPS": 5, "COLOR_AUTO_EXPOSURE": True, "COLOR_EXPOSURE_VAL": 150, "COLOR_AUTO_WB": True,
                "COLOR_WB_VAL": 4600, "COLOR_WEIGHT_THRESHOLD": 20, "QR_SCALE_FACTOR": 1.0, 
                "SAVE_CYCLE": "1_month"
                # ★ CALIB_DATA 기본값 제거됨
            }
            try:
                with open(self.vision_config_file, 'w', encoding='utf-8') as f:
                    json.dump(d, f, indent=4)
            except Exception: pass
            return d
            
        with open(self.vision_config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "SAVE_CYCLE" not in data: data["SAVE_CYCLE"] = "1_month"
            if "COLOR_WEIGHT_THRESHOLD" not in data: data["COLOR_WEIGHT_THRESHOLD"] = 20 
            return data

    # =========================================================================
    # ★ [수정] vision_config가 아닌 product.json의 calibration 하위에 area별로 저장
    # =========================================================================
    def save_calibration_data(self, cell_num, cx, cy, depth, beaker_name):
        try:
            if not beaker_name: return
            if not getattr(self, 'product_config_file', None) or not self.product_config_file.exists(): 
                return
                
            with open(self.product_config_file, 'r', encoding='utf-8') as f:
                p_data = json.load(f)
                
            if "calibration" not in p_data: p_data["calibration"] = {}
            if beaker_name not in p_data["calibration"]: p_data["calibration"][beaker_name] = {}
            
            p_data["calibration"][beaker_name][f"area{cell_num}"] = {
                "cx": cx, 
                "cy": cy, 
                "depth": depth
            }
            
            with open(self.product_config_file, 'w', encoding='utf-8') as f:
                json.dump(p_data, f, indent=4, ensure_ascii=False)
                
            print(f"[VISION] 셀 {cell_num} 교정값({beaker_name}) product.json 저장 완료: X:{cx}, Y:{cy}, Z:{depth:.3f}")
        except Exception as e: 
            print(f"[VISION ERROR] 교정값 저장 실패: {e}")

    def reset_ema(self):
        self.ema_x, self.ema_y, self.ema_r = None, None, None

    def _apply_camera_settings(self, device):
        try:
            dev = device
            auto_exp = self.v_config.get("COLOR_AUTO_EXPOSURE", True)
            dev.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, auto_exp)
            if not auto_exp: dev.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, self.v_config.get("COLOR_EXPOSURE_VAL", 150))
            auto_wb = self.v_config.get("COLOR_AUTO_WB", True)
            dev.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, auto_wb)
            if not auto_wb: dev.set_int_property(OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, self.v_config.get("COLOR_WB_VAL", 4600))
        except Exception: pass

    def detect_circle_by_diff(self, color_img, depth_img, target_color="white"):
        scale = 0.5
        # [원본 보존] 해상도 축소
        small_color = cv2.resize(color_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = small_color.shape[:2]
        
        scale_ratio = w / (self.base_width * scale) 
        min_r = max(10, int(self.v_config["MIN_RADIUS_PX"] * scale_ratio * scale))
        max_r = max(20, int(self.v_config["MAX_RADIUS_PX"] * scale_ratio * scale))
        
        color_threshold = self.v_config.get("COLOR_WEIGHT_THRESHOLD", 20)

        # 1. 색상 맵(Score Map) 생성
        B, G, R = cv2.split(small_color.astype(np.float32))
        if target_color == "white": score_map = B - R
        elif target_color == "black": score_map = 255.0 - ((B + G + R) / 3.0)
        elif target_color == "blue": score_map = B - ((G + R) / 2.0)
        elif target_color == "red": score_map = R - ((B + G) / 2.0)
        elif target_color == "purple": score_map = ((B + R) / 2.0) - G
        elif target_color == "cyan": score_map = ((B + G) / 2.0) - R
        else: score_map = (B + G + R) / 3.0

        # 2. 이진화 및 노이즈 제거
        _, binary = cv2.threshold(np.clip(score_map, 0, 255).astype(np.uint8), color_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 3. 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # =================================================================
        # ★ [핵심 1] 화면에서 가장 큰 색상 덩어리 딱 1개만 남기기
        # 구석에 있는 파란 구조물, 자잘한 반사광 등은 여기서 100% 폐기됩니다.
        # =================================================================
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_cnt = contours[0]
        
        area = cv2.contourArea(largest_cnt)
        if area < (math.pi * (min_r ** 2) * 0.2):
            return None # 뚜껑이 아예 없는 빈 바닥인 경우 방어
            
        # =================================================================
        # ★ [핵심 2] 거리 변환 (Distance Transform)을 이용한 정중앙 탐색
        # 요철(돌기)이 아무리 많아도, 가장 깊숙한 중심점(Core)을 수학적으로 찾아냅니다.
        # =================================================================
        mask_for_dt = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_for_dt, [largest_cnt], -1, 255, -1)
        
        dist_transform = cv2.distanceTransform(mask_for_dt, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        cx, cy = max_loc      # 완벽한 뚜껑의 정중앙 좌표
        inner_radius = max_val # 요철 안쪽의 순수 뚜껑 반경
        
        if not (min_r * 0.5 <= inner_radius <= max_r * 1.5):
            return None
            
        # 요철(돌기)을 포함한 전체 외곽선 크기를 구해 최종 반경을 보정합니다.
        _, _, w_w, w_h = cv2.boundingRect(largest_cnt)
        outer_radius = max(w_w, w_h) / 2.0
        final_radius = (inner_radius + outer_radius) / 2.0 # 중심과 외곽의 평균 크기가 가장 이상적임
        
        # 4. 깊이(Depth) 단차 검사로 최종 승인
        dx, dy, dr = int(cx / scale), int(cy / scale), int(final_radius / scale)
        roi_mask_in = np.zeros(depth_img.shape, dtype=np.uint8)
        cv2.circle(roi_mask_in, (dx, dy), int(dr * 0.7), 255, -1)
        
        roi_mask_out = np.zeros(depth_img.shape, dtype=np.uint8)
        cv2.circle(roi_mask_out, (dx, dy), int(dr * 1.3), 255, -1)
        cv2.circle(roi_mask_out, (dx, dy), int(dr * 1.05), 0, -1)
        
        z_in_vals = depth_img[(roi_mask_in == 255) & (depth_img > 0)]
        z_out_vals = depth_img[(roi_mask_out == 255) & (depth_img > 0)]
        
        if len(z_in_vals) > 50 and len(z_out_vals) > 50:
            z_in = np.median(z_in_vals)
            z_out = np.median(z_out_vals)
            depth_diff = abs(z_out - z_in)
            
            # 단차가 5mm 이상 발생하면 합격!
            if 0.005 <= depth_diff <= 0.45:
                return (dx, dy, dr) 
                
        return None

    def extract_circle_depth(self, cx, cy, r, qr_bbox, depth_aligned):
        if qr_bbox is None: return 0
        pts = np.float32(qr_bbox[0])
        qx, qy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        sf = self.v_config.get("QR_SCALE_FACTOR", 0.4)
        scaled_pts = np.array([[qx + (p[0]-qx)*sf, qy + (p[1]-qy)*sf] for p in pts], dtype=np.int32)
        mask = np.zeros(depth_aligned.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [scaled_pts], 255)
        z_qr = np.median(depth_aligned[(mask == 255) & (depth_aligned > 0)])
        Y, X = np.ogrid[:depth_aligned.shape[0], :depth_aligned.shape[1]]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        annulus = (dist > r + 40) & (dist < r + 100)
        v_f, u_f = np.where((depth_aligned > 0) & annulus)
        if len(v_f) < 50: return 0
        z_f = depth_aligned[v_f, u_f]
        M = np.c_[u_f, v_f, np.ones_like(u_f)]
        C, _, _, _ = np.linalg.lstsq(M, z_f, rcond=None)
        return float((C[0]*cx + C[1]*cy + C[2]) - ((C[0]*qx + C[1]*qy + C[2]) - z_qr))

    def measure_beaker_and_qr(self, beaker_name="", is_calib=False):
        self.v_config = self.load_vision_config()
        target_color = "white" 
        
        if beaker_name:
            try:
                if hasattr(self, 'product_config_file') and self.product_config_file.exists():
                    with open(self.product_config_file, 'r', encoding='utf-8') as f:
                        p_data = json.load(f)
                        
                    if is_calib:
                        b_data = p_data.get("calibration", {}).get(beaker_name, {})
                        target_mm = b_data.get("anchor_mm", b_data.get("target_mm", 150.0))
                    else:
                        b_data = p_data.get("beaker", {}).get(beaker_name, {})
                        target_mm = b_data.get("target_mm", 150.0)
                    
                    target_color = b_data.get("color", "white")

                    if target_mm > 0:
                        ratio = target_mm / 150.0
                        self.v_config["MIN_RADIUS_PX"] = int(300 * ratio)
                        self.v_config["MAX_RADIUS_PX"] = int(800 * ratio)
            except Exception: pass
        
        if self.device is None:
            print("[VISION] 카메라가 연결되어 있지 않아 연결을 시도합니다.")
            self.connect_camera()
            if self.device is None: 
                return -1, -1, 0, -1.0, "FAIL"

        self.reset_ema()
        z_history = []
        valid_count = 0
        pipeline = None

        try:
            print("[VISION] 로봇 이동 완료 확인. 비전 파이프라인 시작 (촬영 준비)")
            pipeline = Pipeline(self.device)
            cfg = Config()
            c_prof = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(
                self.v_config["COLOR_WIDTH"], self.v_config["COLOR_HEIGHT"], OBFormat.RGB, self.v_config["COLOR_FPS"])
            d_prof = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
                self.v_config["DEPTH_WIDTH"], self.v_config["DEPTH_HEIGHT"], OBFormat.Y16, self.v_config["DEPTH_FPS"])
            cfg.enable_stream(c_prof)
            cfg.enable_stream(d_prof)
            cfg.set_align_mode(OBAlignMode.HW_MODE)
            
            pipeline.start(cfg)

            print("[VISION] 렌즈 안정화 및 잔상 제거 중 (1.5초 대기)...")
            start_time = time.time()
            while time.time() - start_time < 1.5:
                try: pipeline.wait_for_frames(100) 
                except: pass

            last_qr_data = "FAIL"
            last_qr_pts = None

            print("[VISION] 정밀 촬영 시작!")
            
            for attempt in range(30):
                frames = None
                try: 
                    frames = pipeline.wait_for_frames(1000) 
                except: 
                    print(f"  - 프레임 대기 타임아웃 ({attempt+1}/30)")
                    continue
                    
                if not frames: continue

                cf, df = frames.get_color_frame(), frames.get_depth_frame()
                if not cf or not df: continue

                color_img = cv2.cvtColor(np.frombuffer(cf.get_data(), dtype=np.uint8).reshape((cf.get_height(), cf.get_width(), 3)), cv2.COLOR_RGB2BGR)
                depth_img = np.frombuffer(df.get_data(), dtype=np.uint16).reshape((df.get_height(), df.get_width())) * 0.001 
                
                if np.count_nonzero(depth_img) < 1000:
                    print(f"  - Depth 데이터 안정화 대기 중... ({attempt+1}/30)")
                    continue

                display_img = color_img.copy()
                h, w = display_img.shape[:2]
                l_thick = max(3, int(w / 800))
                cam_cx, cam_cy = w // 2, h // 2
                
                cv2.line(display_img, (cam_cx - 60, cam_cy), (cam_cx + 60, cam_cy), (0, 255, 255), l_thick)
                cv2.line(display_img, (cam_cx, cam_cy - 60), (cam_cx, cam_cy + 60), (0, 255, 255), l_thick)

                barcodes = self._read_qr_robust(color_img)
                circle = self.detect_circle_by_diff(color_img, depth_img, target_color)

                if barcodes:
                    last_qr_data = barcodes[0].text
                    last_qr_pts = np.array([[p.x, p.y] for p in [barcodes[0].position.top_left, barcodes[0].position.top_right, barcodes[0].position.bottom_right, barcodes[0].position.bottom_left]], dtype=np.int32)
                    cv2.polylines(display_img, [last_qr_pts], isClosed=True, color=(255, 0, 255), thickness=l_thick)

                if circle:
                    cx, cy, cr = circle
                    if self.ema_x is None: 
                        self.ema_x, self.ema_y, self.ema_r = cx, cy, cr
                    else:
                        self.ema_x = self.ALPHA * cx + (1 - self.ALPHA) * self.ema_x
                        self.ema_y = self.ALPHA * cy + (1 - self.ALPHA) * self.ema_y
                        self.ema_r = self.ALPHA * cr + (1 - self.ALPHA) * self.ema_r
                        
                    ex, ey, er = int(self.ema_x), int(self.ema_y), int(self.ema_r)

                    cv2.circle(display_img, (ex, ey), er, (0, 255, 0), l_thick)
                    cv2.line(display_img, (ex - 50, ey), (ex + 50, ey), (0, 0, 255), l_thick)
                    cv2.line(display_img, (ex, ey - 50), (ex, ey + 50), (0, 0, 255), l_thick)
                    
                    if last_qr_pts is not None:
                        qx, qy = np.mean(last_qr_pts[:, 0]), np.mean(last_qr_pts[:, 1])
                        if math.sqrt((self.ema_x - qx)**2 + (self.ema_y - qy)**2) <= self.ema_r:
                            cz = self.extract_circle_depth(self.ema_x, self.ema_y, self.ema_r, [last_qr_pts], depth_img)
                            if cz > 0: 
                                z_history.append(cz)
                                valid_count += 1

                text_size = max(2.0, 4 * (w / 3840.0))
                if self.ema_x is not None:
                    live_z = float(np.median(z_history)) + self.v_config["Z_OFFSET_METERS"] if z_history else 0.0
                    cv2.putText(display_img, f"X: {int(self.ema_x)}  Y: {int(self.ema_y)}  Z: {live_z*1000:.1f}mm", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), l_thick + 1)
                
                cv2.putText(display_img, f"QR: {last_qr_data}", (50, h - 100), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 255), l_thick + 1)

                if self.ema_x is not None and last_qr_data != "FAIL" and valid_count > 0:
                    print(f"[VISION] ➔ 측정 성공! (시도 {attempt+1}회 만에 검출)")
                    if self.on_image_ready: 
                        self.on_image_ready(display_img)
                    break

            if self.ema_x and valid_count > 0 and z_history:
                z_res = float(np.median(z_history)) + self.v_config["Z_OFFSET_METERS"]
                return int(self.ema_x), int(self.ema_y), int(self.ema_r), z_res, last_qr_data
            
            print("[VISION WARN] 비커 원 검출 실패")
            if self.on_image_ready and display_img is not None:
                self.on_image_ready(display_img)
                
            return -1, -1, 0, -1.0, last_qr_data

        except Exception as e:
            print(f"[VISION ERROR] 측정 중 예외 발생: {e}")
            return -1, -1, 0, -1.0, "ERROR"
            
        finally:
            print("[VISION] 카메라 파이프라인 종료 (렌즈 OFF)")
            if pipeline: 
                try: pipeline.stop()
                except: pass

    def _read_qr_robust(self, color_img):
        target_formats = zxingcpp.BarcodeFormat.QRCode | zxingcpp.BarcodeFormat.DataMatrix | zxingcpp.BarcodeFormat.MicroQRCode

        h, w = color_img.shape[:2]
        
        roi_w, roi_h = int(w * 0.40), int(h * 0.50)  # 가로 40%, 세로 50% 중앙 박스
        start_x, start_y = (w - roi_w) // 2, (h - roi_h) // 2
        
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        
        # 화면 크기와 동일한 까만 도화지를 만들고, 가운데 구멍만 뚫어줍니다.
        mask = np.zeros_like(gray)
        cv2.rectangle(mask, (start_x, start_y), (start_x + roi_w, start_y + roi_h), 255, -1)
        
        # 원본 이미지에 까만 도화지를 덮어씌워 주변 구리벽을 완전히 암전시킵니다.
        safe_gray = cv2.bitwise_and(gray, mask)

        # 1. 그레이스케일 탐색 (가장 빠름)
        barcodes = zxingcpp.read_barcodes(safe_gray, formats=target_formats)
        if barcodes: return barcodes

        # 2. 감마 보정 (Gamma Correction)
        invGamma = 1.0 / 0.5  
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        dark_gray = cv2.LUT(safe_gray, table)
        barcodes = zxingcpp.read_barcodes(dark_gray, formats=target_formats)
        if barcodes: return barcodes

        # 3. CLAHE (국소 대조비 극대화)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(safe_gray)
        barcodes = zxingcpp.read_barcodes(enhanced, formats=target_formats)
        if barcodes: return barcodes

        # 4. 적응형 이진화 (Adaptive Threshold)
        adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)
        # 필터 적용 시 마스크 경계선에 생기는 하얀 찌꺼기를 다시 한번 잘라냅니다.
        adaptive_thresh = cv2.bitwise_and(adaptive_thresh, mask) 
        barcodes = zxingcpp.read_barcodes(adaptive_thresh, formats=target_formats)
        if barcodes: return barcodes

        # 5. Blur + Otsu 이진화
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, mask)
        barcodes = zxingcpp.read_barcodes(binary, formats=target_formats)
        if barcodes: return barcodes

        # 6. 모폴로지 닫힘 (Morphology Close)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        closed_binary = cv2.bitwise_and(closed_binary, mask)
        barcodes = zxingcpp.read_barcodes(closed_binary, formats=target_formats)
        if barcodes: return barcodes

        # 7. OpenCV 기본 디텍터 최후의 보루
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(safe_gray)
        if data and bbox is not None and len(data) > 0:
            class DummyPoint:
                def __init__(self, x, y): self.x, self.y = int(x), int(y)
            class DummyPos:
                def __init__(self, pts):
                    self.top_left = DummyPoint(pts[0][0], pts[0][1])
                    self.top_right = DummyPoint(pts[1][0], pts[1][1])
                    self.bottom_right = DummyPoint(pts[2][0], pts[2][1])
                    self.bottom_left = DummyPoint(pts[3][0], pts[3][1])
            class DummyBarcode:
                def __init__(self, text, pts):
                    self.text = text
                    self.position = DummyPos(pts)
            return [DummyBarcode(data, bbox[0])]

        return None
                
# # =========================================================================
# # [독립 테스트 코드] 비커 원 검출 및 QR 인식 개별 실행 모듈
# # =========================================================================
# if __name__ == "__main__":
#     def run_vision_test():
#         print("=== 비전 단독 테스트 모드 시작 ===")
#         vision = VisionCore()
        
#         # 1. 카메라 연결 테스트
#         print("\n1. 카메라 연결 시도...")
#         if not vision.connect_camera():
#             print("[에러] 카메라 연결 실패로 테스트를 종료합니다.")
#             return
            
#         print("\n2. 파이프라인 구동 및 스트림 캡처 준비...")
#         pipeline = None
#         try:
#             pipeline = Pipeline(vision.device)
#             cfg = Config()
#             c_prof = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(
#                 vision.v_config["COLOR_WIDTH"], vision.v_config["COLOR_HEIGHT"], OBFormat.RGB, vision.v_config["COLOR_FPS"])
#             d_prof = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
#                 vision.v_config["DEPTH_WIDTH"], vision.v_config["DEPTH_HEIGHT"], OBFormat.Y16, vision.v_config["DEPTH_FPS"])
#             cfg.enable_stream(c_prof)
#             cfg.enable_stream(d_prof)
#             cfg.set_align_mode(OBAlignMode.HW_MODE)
#             pipeline.start(cfg)

#             print(" - 하드웨어 안정화 대기 (2초)...")
#             for _ in range(30):
#                 try: pipeline.wait_for_frames(100)
#                 except: pass

#             print("\n3. 이미지 캡처 및 개별 모듈 테스트 시작")
#             frames = None
#             for _ in range(30): # 최대 5번 캡처 시도
#                 try:
#                     frames = pipeline.wait_for_frames(1000)
#                     if frames and frames.get_color_frame() and frames.get_depth_frame():
#                         break
#                 except Exception as e:
#                     print(f" - 프레임 캡처 대기... {e}")
                    
#             if not frames:
#                 print("[에러] 유효한 프레임을 캡처하지 못했습니다.")
#                 return

#             cf = frames.get_color_frame()
#             df = frames.get_depth_frame()
            
#             color_img = cv2.cvtColor(np.frombuffer(cf.get_data(), dtype=np.uint8).reshape((cf.get_height(), cf.get_width(), 3)), cv2.COLOR_RGB2BGR)
#             depth_img = np.frombuffer(df.get_data(), dtype=np.uint16).reshape((df.get_height(), df.get_width())) * 0.001
#             display_img = color_img.copy()

#             print("\n--- [A] QR 코드 인식 모듈 테스트 ---")
#             t1 = time.time()
#             barcodes = zxingcpp.read_barcodes(color_img)
#             t2 = time.time()
#             if barcodes:
#                 qr_text = barcodes[0].text
#                 qr_pts = np.array([[p.x, p.y] for p in [barcodes[0].position.top_left, barcodes[0].position.top_right, barcodes[0].position.bottom_right, barcodes[0].position.bottom_left]], dtype=np.int32)
#                 cv2.polylines(display_img, [qr_pts], isClosed=True, color=(255, 0, 255), thickness=8)
#                 print(f" ✅ QR 검출 성공: '{qr_text}' (소요시간: {(t2-t1)*1000:.1f}ms)")
#             else:
#                 print(" ❌ QR 검출 실패")

#             print("\n--- [B] 비커 외곽(원) 검출 모듈 테스트 ---")
#             t1 = time.time()
#             target_color = "white"
#             circle = vision.detect_circle_by_diff(color_img, depth_img, target_color)
#             t2 = time.time()
#             if circle:
#                 cx, cy, cr = circle
#                 cv2.circle(display_img, (cx, cy), cr, (0, 255, 0), 8)
#                 cv2.circle(display_img, (cx, cy), 15, (0, 0, 255), -1)
#                 print(f" ✅ 원 검출 성공: 중심(X:{cx}, Y:{cy}), 반경(R:{cr}) (소요시간: {(t2-t1)*1000:.1f}ms)")
                
#                 if barcodes:
#                     print("\n--- [C] 깊이(Z) 추출 연산 테스트 ---")
#                     try:
#                         cz = vision.extract_circle_depth(cx, cy, cr, [qr_pts], depth_img)
#                         print(f" ✅ 깊이 연산 성공: {cz:.4f} m (오프셋 미적용 Raw Data)")
#                     except Exception as e:
#                         print(f" ❌ 깊이 연산 에러: {e}")
#             else:
#                 print(" ❌ 원 검출 실패")

#             # 결과 화면 출력 (ESC를 누르면 종료)
#             small_display = cv2.resize(display_img, (1280, 720))
#             cv2.imshow("Vision Standalone Test", small_display)
#             print("\n테스트가 완료되었습니다. 결과 창에서 아무 키나 누르면 종료됩니다.")
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

#         except Exception as e:
#             print(f"[테스트 에러] {e}")
#         finally:
#             if pipeline:
#                 pipeline.stop()
#             print("=== 비전 테스트 종료 ===")

#     # 터미널에서 이 파일을 직접 실행(python vision_main.py)했을 때만 작동
#     run_vision_test()
