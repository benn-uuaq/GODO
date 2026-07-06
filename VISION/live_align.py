import cv2
import numpy as np
import sys
import time
from pathlib import Path
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode

# VISION 폴더 안에서 실행할 때 경로 문제 해결
curr_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(curr_dir.parent))

from vision_main import VisionCore

def run_live_alignment():
    print("==================================================")
    print(" 🎯 로봇 티칭용 실시간 비커 정렬 뷰어 (렉 방지 버전)")
    print("==================================================")
    
    core = VisionCore()
    if not core.connect_camera():
        print("[ERROR] 카메라 연결 실패!")
        return

    pipeline = Pipeline(core.device)
    cfg = Config()
    
    # 설정 로드
    c_w, c_h = core.v_config["COLOR_WIDTH"], core.v_config["COLOR_HEIGHT"]
    d_w, d_h = core.v_config["DEPTH_WIDTH"], core.v_config["DEPTH_HEIGHT"]
    fps = core.v_config["COLOR_FPS"]

    c_prof = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(c_w, c_h, OBFormat.RGB, fps)
    d_prof = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(d_w, d_h, OBFormat.Y16, fps)
    
    cfg.enable_stream(c_prof)
    cfg.enable_stream(d_prof)
    cfg.set_align_mode(OBAlignMode.HW_MODE)
    pipeline.start(cfg)

    window_name = "Live Alignment (Q to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("[INFO] 스트리밍 시작. 'q'를 누르면 종료됩니다.")

    try:
        while True:
            start_time = time.time() # 성능 체크용

            try:
                frames = pipeline.wait_for_frames(200) # 대기 시간 최적화
                if not frames: continue
            except: continue
            
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df: continue

            # 1. 원본 데이터 변환 (정밀 연산용)
            color_img = cv2.cvtColor(np.frombuffer(cf.get_data(), dtype=np.uint8).reshape((cf.get_height(), cf.get_width(), 3)), cv2.COLOR_RGB2BGR)
            depth_img = np.frombuffer(df.get_data(), dtype=np.uint16).reshape((df.get_height(), df.get_width())) * 0.001
            
            # ★ 핵심 렉 방지: 화면 출력용 이미지는 절반 크기(0.5배)로 축소하여 생성
            display_img = cv2.resize(color_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            h, w = display_img.shape[:2]
            
            # 축소된 이미지의 중심 계산
            cam_cx, cam_cy = w // 2, h // 2
            
            # 2. 비전 연산 (정밀도는 유지하기 위해 원본 이미지 사용)
            circle = core.detect_circle_by_diff(color_img, depth_img)
            
            # [그리기 1] 카메라 정중앙 가이드 (축소 이미지 기준)
            cv2.line(display_img, (cam_cx - 80, cam_cy), (cam_cx + 80, cam_cy), (0, 255, 255), 2)
            cv2.line(display_img, (cam_cx, cam_cy - 80), (cam_cx, cam_cy + 80), (0, 255, 255), 2)
            cv2.putText(display_img, "CENTER", (cam_cx + 10, cam_cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if circle:
                # 비전 결과값도 0.5배 축소하여 화면에 매칭
                bx, by, br = [int(v * 0.5) for v in circle]
                
                # [그리기 2] 비커 중심 표시
                cv2.circle(display_img, (bx, by), br, (0, 255, 0), 2)
                cv2.line(display_img, (bx - 40, by), (bx + 40, by), (0, 255, 0), 2)
                cv2.line(display_img, (bx, by - 40), (bx, by + 40), (0, 255, 0), 2)
                
                # 오프셋 선 (원본 해상도 차이를 고려하여 픽셀값은 원본 기준으로 표시)
                orig_dx = int(circle[0] - (c_w // 2))
                orig_dy = int(circle[1] - (c_h // 2))
                
                cv2.putText(display_img, f"BEAKER FOUND", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_img, f"Offset X: {orig_dx} px", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display_img, f"Offset Y: {orig_dy} px", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if abs(orig_dx) <= 10 and abs(orig_dy) <= 10:
                    cv2.putText(display_img, "ALIGNMENT OK!", (w//2 - 100, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(display_img, "Searching...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            # 3. 화면 출력 (가벼운 이미지 송출)
            cv2.imshow(window_name, display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt: pass
    finally:
        if pipeline:
            try: pipeline.stop()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_alignment()