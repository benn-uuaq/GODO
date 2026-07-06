import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import ast
import sys
import subprocess
import json
import platform
import time
import datetime
import traceback
import threading
import ctypes
import sqlite3
import socket
import csv
import math
import queue
from pathlib import Path

from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.QtGui import QBrush, QColor, QFont, QImage
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal, QDateTime, pyqtSlot, QThread

sys.setswitchinterval(0.001)

def get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).parent
        internal_dir = base_dir / "_internal"
        if internal_dir.exists() and (internal_dir / "DB").exists():
            return internal_dir
        return base_dir
    else:
        return Path(__file__).resolve().parent

APP_ROOT = get_app_root()
sys.path.insert(0, str(APP_ROOT))

from UI.godo_ui import Ui_MainWindow
from ROBOT.robot import Robot_29999, Robot_30001, AlarmManager, Robot_modbus
from DB.db import RobotDB
from VISION.vision_main import VisionCore

LOGO_PATH = APP_ROOT / "logo.png"
DB_PATH = APP_ROOT / "DB"
CONFIG_PATH = DB_PATH / "config"
VISION_PATH = DB_PATH / "vision"
PRODUCT_PATH = DB_PATH / "product"
ROBOT_PATH = DB_PATH / "robot"

# print(f"OpenCV version: {cv2.__version__}")
# print(f"Python version: {sys.version}")

# =========================================================================
# [클래스] Print 출력 리다이렉트 (시스템 메시지용)
# =========================================================================
class EmittingStream(QtCore.QObject):
    textWritten = pyqtSignal(str)
    
    def write(self, text):
        if text.strip():  # 빈 줄바꿈 무시
            self.textWritten.emit(str(text))
            
    def flush(self):
        pass
    
# =========================================================================
# [클래스 1] 완벽히 독립된 1단계: 하드웨어 전용 백그라운드 OS 스레드
# =========================================================================
class VisionHardwareWorker(threading.Thread):
    def __init__(self, config_path, product_path, req_queue, res_queue, img_queue):
        super().__init__(daemon=True) 
        self.config_path = config_path
        self.product_path = product_path 
        self.req_queue = req_queue
        self.res_queue = res_queue
        self.img_queue = img_queue
        self.last_img = None # ★ 촬영된 원본 이미지를 캐싱할 변수

    def run(self):
        from VISION.vision_main import VisionCore
        self.core = VisionCore(config_path=self.config_path, product_path=self.product_path)

        def _on_image(np_img):
            self.last_img = np_img # ★ 비전 파이프라인에서 그려진 최종 결과 이미지를 들고 있음
            if not self.img_queue.full():
                self.img_queue.put(np_img)

        self.core.on_image_ready = _on_image
        
        is_connected = self.core.connect_camera()
        self.res_queue.put({"type": "status", "ready": self.core.device is not None})

        while True:
            req = self.req_queue.get() 
            if req is None: break 

            area = req['area']
            is_calib = req['is_calib']
            beaker_name = req.get('beaker_name', "")
            jig_num = req.get('jig_num', 0) # ★ 지그 번호 수신

            self.last_img = None # 새로운 촬영을 위해 초기화

            try:
                cx, cy, cr, depth, qr = self.core.measure_beaker_and_qr(beaker_name, is_calib)
                
                # =========================================================
                # ★ [추가] 검사 이미지(Result) 로컬 폴더 저장 로직
                # =========================================================
                if not is_calib and self.last_img is not None:
                    try:
                        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_qr = str(qr).replace("/", "_").replace("\\", "_").replace(":", "") # 파일명 오류 방지
                        
                        # 파일명: 날짜_비커이름_아레아번호_지그번호_QR번호.jpg
                        filename = f"{now_str}_{beaker_name}_Area{area}_Jig{jig_num}_QR_{safe_qr}.jpg"
                        
                        save_dir = DB_PATH / "Log" / "Image"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        filepath = save_dir / filename
                        
                        cv2.imwrite(str(filepath), self.last_img)
                        print(f"[VISION LOG] 검사 이미지 저장 완료: {filename}")
                    except Exception as e:
                        print(f"[VISION LOG ERROR] 이미지 저장 실패: {e}")

                self.res_queue.put({"type": "result", "area": area, "cx": cx, "cy": cy, "cr": cr, "depth": depth, "qr": qr, "is_calib": is_calib})
            except Exception as e:
                print(f"[HW THREAD ERROR] {e}")
                self.res_queue.put({"type": "result", "area": area, "cx": -1, "cy": -1, "cr": 0, "depth": 0.0, "qr": "FAIL", "is_calib": is_calib})

# =========================================================================
# [클래스 2] 중간 다리 2단계: 큐(Queue) 데이터를 읽어 GUI로 뿌려주는 QThread
# =========================================================================
class VisionBridgeQThread(QThread):
    result_signal = pyqtSignal(int, float, float, float, float, str, bool)
    image_signal = pyqtSignal(QImage)
    ready_signal = pyqtSignal(bool)

    def __init__(self, config_path, product_path=None):
        super().__init__()
        self.req_queue = queue.Queue()
        self.res_queue = queue.Queue()
        self.img_queue = queue.Queue(maxsize=2) # 메모리 폭발 방지용 이미지 2장 제한
        
        # 1단계 하드웨어 스레드를 품고 있습니다.
        self.hw_worker = VisionHardwareWorker(config_path, product_path, self.req_queue, self.res_queue, self.img_queue) 
        self.running = True

    def run(self):
        self.hw_worker.start() # 하드웨어 스레드 구동 시작

        while self.running:
            # 1. 결과 수신 처리 (비동기 확인)
            try:
                while True:
                    res = self.res_queue.get_nowait()
                    if res["type"] == "status":
                        self.ready_signal.emit(res["ready"])
                    elif res["type"] == "result":
                        self.result_signal.emit(res["area"], res["cx"], res["cy"], res["cr"], res["depth"], res["qr"], res["is_calib"])
            except queue.Empty:
                pass

            # 2. 이미지 수신 및 UI 변환 (이 작업만 QThread에서 안전하게 수행)
            try:
                while True:
                    np_img = self.img_queue.get_nowait()
                    # 순수 Numpy 이미지를 PyQt 전용 QImage로 변환
                    resized = cv2.resize(np_img, (640, 360), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                    self.image_signal.emit(qimg)
            except queue.Empty:
                pass

            time.sleep(0.02) # 중간 다리의 부하를 막기 위한 아주 짧은 숨구멍

    def request_task(self, area, is_calib=False, beaker_name="", jig_num=0):
        """메인 프로그램(GUI)이 비전 측정을 요청할 때 호출하는 입구"""
        self.req_queue.put({"area": area, "is_calib": is_calib, "beaker_name": beaker_name, "jig_num": jig_num})

    def stop(self):
        self.running = False
        self.req_queue.put(None) # 하드웨어 스레드 강제 종료 신호
        self.quit()
        self.wait()

# =========================================================================
# [클래스] 외부 통신 (TCP Server) 쓰레드
# =========================================================================
class ExternalTcpServerThread(QThread):
    def __init__(self, main_window, port=5000):
        super().__init__()
        self.main = main_window
        self.port = port
        self.running = True
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def run(self):
        try:
            self.server_sock.bind(("0.0.0.0", self.port))
            self.server_sock.listen(5)
            self.server_sock.settimeout(1.0)
            print(f"[System] 외부 통신 TCP 서버 시작됨 (포트: {self.port})...")
            
            while self.running:
                try:
                    client_sock, addr = self.server_sock.accept()
                    # ★ 서브 스레드로 던지지 않고 직렬(동기) 처리하여 최우선 순위(HighestPriority)를 보장받음
                    self.handle_client(client_sock, addr) 
                except socket.timeout:
                    continue
        except Exception as e:
            print(f"[TCP 서버 에러] {e}")

    def handle_client(self, client_sock, addr):
        try:
            # ★ 카메라 4K 연산(최대 5~7초)을 버틸 수 있도록 서버 측 수신 타임아웃을 넉넉히 15초로 늘림
            client_sock.settimeout(15.0) 
            
            data = client_sock.recv(2048)
            if data:
                req_str = data.decode('utf-8').strip()
                self.main.save_tcp_csv(f"📥 [수신] {req_str}")
                
                try:
                    req_json = json.loads(req_str)
                    response = self.main.process_external_request(req_json)
                    
                    resp_str = json.dumps(response, ensure_ascii=False) + "\n"
                    client_sock.sendall(resp_str.encode('utf-8'))
                    self.main.save_tcp_csv(f"📤 [송신] {resp_str.strip()}")
                except json.JSONDecodeError:
                    err = json.dumps({"area": 0, "result": "fail", "result_code": 2, "msg": "JSON 파싱 오류"}) + "\n"
                    client_sock.sendall(err.encode('utf-8'))
                    self.main.save_tcp_csv(f"📤 [송신] {err.strip()}")
        except socket.timeout:
            print(f"[TCP 에러] 클라이언트({addr}) 타임아웃 - 응답 대기 시간 초과")
        except Exception as e:
            print(f"[TCP 클라이언트 에러] {e}")
        finally:
            client_sock.close()

    def stop(self):
        self.running = False
        try: self.server_sock.close()
        except: pass
        self.quit()
        self.wait()

# =========================================================================
# [클래스] DB 관리 (전면 개편: 24개 고정 슬롯 방식)
# =========================================================================
class DBManager(RobotDB):
    def __init__(self):
        super().__init__()
        # 아카이브 저장 폴더 경로 생성
        self.archive_dir = APP_ROOT / "DB" / "Log" / "Result"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def archive_current_results(self):
        """현재 DB의 result 테이블 내용을 기본 csv 모듈을 사용해 아카이빙합니다."""
        try:
            db_files = list((APP_ROOT / "DB").glob("*.db"))
            if not db_files: return
            
            conn = sqlite3.connect(db_files[0])
            cursor = conn.cursor()
            
            # 현재 result 테이블의 데이터를 가져옴 (데이터가 있는 경우만)
            cursor.execute("SELECT * FROM result WHERE work_state != '-'")
            rows = cursor.fetchall()
            
            if rows:
                # 컬럼 이름(헤더) 가져오기
                column_names = [description[0] for description in cursor.description]
                
                # 파일명 생성: WorkResult_YYYYMMDD_HHMMSS.csv
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"WorkResult_{timestamp}.csv"
                file_path = self.archive_dir / file_name
                
                # CSV 파일로 저장 (내장 라이브러리 사용, 한글 깨짐 방지 utf-8-sig)
                with open(file_path, mode='w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(column_names)  # 헤더(열 이름) 쓰기
                    writer.writerows(rows)         # 데이터 쓰기
                    
                print(f"[SYSTEM] 이전 작업 기록 아카이브 완료: {file_name}")
            else:
                print("[SYSTEM] 저장할 이전 작업 데이터가 없어 아카이브를 건너뜁니다.")
                
            conn.close()
                
        except Exception as e:
            print(f"[DB ERROR] 아카이브 생성 실패: {e}")
        
    def get_work_goals(self):
        try:
            db_dir = APP_ROOT / "DB"
            db_files = list(db_dir.glob("*.db"))
            if not db_files: return {}
            
            db_path = db_files[0]
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("PRAGMA table_info(work_goal)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if "beaker_type" not in columns:
                cursor.execute("ALTER TABLE work_goal ADD COLUMN beaker_type TEXT DEFAULT ''")
            if "work_mode" not in columns:
                cursor.execute("ALTER TABLE work_goal ADD COLUMN work_mode TEXT DEFAULT '순차 작업'")
            if "target_points" not in columns:
                cursor.execute("ALTER TABLE work_goal ADD COLUMN target_points TEXT DEFAULT '[]'")
            if "done_points" not in columns:
                cursor.execute("ALTER TABLE work_goal ADD COLUMN done_points TEXT DEFAULT '[]'")
            conn.commit()
            
            cursor.execute("SELECT work_cell, work_count, beaker_type, work_mode, target_points, done_points FROM work_goal")
            rows = cursor.fetchall()
            conn.close()
            
            result = {}
            for r in rows:
                try: t_pts = json.loads(r[4]) if r[4] else []
                except: t_pts = []
                try: d_pts = json.loads(r[5]) if r[5] else []
                except: d_pts = []
                
                result[str(r[0])] = {
                    "count": r[1], 
                    "beaker": r[2] if r[2] else "",
                    "work_mode": r[3] if r[3] else "순차 작업",
                    "target_points": t_pts,
                    "done_points": d_pts
                }
            return result
        except Exception as e:
            print(f"[DB ERROR] 작업 목표 불러오기 실패: {e}")
            return {}

    def get_work_done_counts(self):
        try:
            db_dir = APP_ROOT / "DB"
            db_files = list(db_dir.glob("*.db"))
            if not db_files: return {}
            
            db_path = db_files[0]
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT work_cell, work_count FROM work_done")
            rows = cursor.fetchall()
            conn.close()
            
            result = {}
            for r in rows:
                result[str(r[0])] = r[1]
            return result
        except Exception:
            return {}

    def init_result_table(self):
        db_dir = APP_ROOT / "DB"
        db_files = list(db_dir.glob("*.db"))
        if not db_files: return
        db_path = db_files[0]
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS result (
                no INTEGER PRIMARY KEY,
                work_cell INTEGER,
                work_point INTEGER,
                beaker_type TEXT,
                qr_value TEXT,
                robot_pose TEXT,
                calib_loc TEXT,
                beaker_loc TEXT,
                loc_deviation TEXT,
                calib_depth TEXT,
                beaker_depth TEXT,
                depth_deviation TEXT,
                work_state TEXT,
                start_time TEXT,
                end_time TEXT
            )
        """)
        
        cursor.execute("SELECT COUNT(*) FROM result")
        if cursor.fetchone()[0] != 24:
            cursor.execute("DELETE FROM result") 
            for c in [1, 2]:
                for p in range(1, 13):
                    no = (c - 1) * 12 + p  
                    cursor.execute("""
                        INSERT INTO result 
                        (no, work_cell, work_point, beaker_type, qr_value, robot_pose, calib_loc, beaker_loc, loc_deviation, calib_depth, beaker_depth, depth_deviation, work_state, start_time, end_time)
                        VALUES (?, ?, ?, '', '', '', '', '', '', '', '', '', '-', '', '')
                    """, (no, c, p))
        conn.commit()
        conn.close()
        # ★ 누락되었던 엑셀 내보내기 코드 추가
        self.export_to_excel()

    def setup_new_work_by_cell(self, cell, count, beaker_type="", work_mode="순차 작업", target_points=None):
        if target_points is None: target_points = []
        
        self.archive_current_results()
        
        db_files = list((APP_ROOT / "DB").glob("*.db"))
        if not db_files: return
        conn = sqlite3.connect(db_files[0])
        cursor = conn.cursor()
        
        t_pts_str = json.dumps(target_points)
        d_pts_str = json.dumps([])
        
        cursor.execute("SELECT COUNT(*) FROM work_goal WHERE work_cell=?", (cell,))
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO work_goal (work_cell, work_count, beaker_type, work_mode, target_points, done_points) VALUES (?, ?, ?, ?, ?, ?)",
                           (cell, count, beaker_type, work_mode, t_pts_str, d_pts_str))
        else:
            cursor.execute("""
                UPDATE work_goal 
                SET work_count=?, beaker_type=?, work_mode=?, target_points=?, done_points=?
                WHERE work_cell=?
            """, (count, beaker_type, work_mode, t_pts_str, d_pts_str, cell))
            
        cursor.execute("""
            UPDATE result 
            SET beaker_type=?, qr_value='', robot_pose='', 
                calib_loc='', beaker_loc='', loc_deviation='', 
                calib_depth='', beaker_depth='', depth_deviation='', 
                work_state='-', start_time='', end_time=''
            WHERE work_cell=?
        """, (beaker_type, int(cell)))
        
        valid_pts = set(target_points) if target_points else set()
        for pt in valid_pts:
            cursor.execute("UPDATE result SET work_state='waiting' WHERE work_cell=? AND work_point=?", (int(cell), int(pt)))
                
        conn.commit()
        conn.close()
        self.export_to_excel()

    def update_work_done_points(self, cell, done_points):
        db_files = list((APP_ROOT / "DB").glob("*.db"))
        if not db_files: return
        conn = sqlite3.connect(db_files[0])
        cursor = conn.cursor()
        d_pts_str = json.dumps(done_points)
        cursor.execute("UPDATE work_goal SET done_points=? WHERE work_cell=?", (d_pts_str, cell))
        conn.commit()
        conn.close()
        self.export_to_excel()

    def insert_new_job(self, cell, point, beaker_type):
        db_files = list((APP_ROOT / "DB").glob("*.db"))
        if not db_files: return
        conn = sqlite3.connect(db_files[0])
        cursor = conn.cursor()
        start_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        
        cursor.execute("""
            UPDATE result 
            SET beaker_type=?, work_state='진행중', start_time=?
            WHERE work_cell=? AND work_point=?
        """, (beaker_type, start_time, int(cell), int(point)))
        conn.commit()
        conn.close()
        # ★ 누락되었던 엑셀 내보내기 코드 추가
        self.export_to_excel()

    def update_job_vision_data(self, cell, point, qr, r_pose, c_loc, b_loc, loc_dev, c_depth, b_depth, depth_dev):
        db_files = list((APP_ROOT / "DB").glob("*.db"))
        if not db_files: return
        conn = sqlite3.connect(db_files[0])
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE result 
            SET qr_value=?, robot_pose=?, calib_loc=?, beaker_loc=?, loc_deviation=?, 
                calib_depth=?, beaker_depth=?, depth_deviation=?
            WHERE work_cell=? AND work_point=? AND work_state='진행중'
        """, (qr, r_pose, c_loc, b_loc, loc_dev, c_depth, b_depth, depth_dev, int(cell), int(point)))
        conn.commit()
        conn.close()
        # ★ 누락되었던 엑셀 내보내기 코드 추가
        self.export_to_excel()

    def finish_job(self, cell, point):
        db_files = list((APP_ROOT / "DB").glob("*.db"))
        if not db_files: return
        conn = sqlite3.connect(db_files[0])
        cursor = conn.cursor()
        end_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        cursor.execute("""
            UPDATE result 
            SET work_state='완료', end_time=?
            WHERE work_cell=? AND work_point=? AND work_state='진행중'
        """, (end_time, int(cell), int(point)))
        conn.commit()
        conn.close()
        # ★ 누락되었던 엑셀 내보내기 코드 추가
        self.export_to_excel()
        
    def update_calibration_data_all_points(self, cell, loc, depth):
        """해당 셀의 모든 결과 데이터에 교정 좌표와 깊이를 일괄 저장합니다."""
        db_files = list((APP_ROOT / "DB").glob("*.db"))
        if not db_files: return
        try:
            conn = sqlite3.connect(db_files[0])
            cursor = conn.cursor()
            # 해당 셀(cell)에 속하는 1~12번 포인트를 모두 찾아 교정 데이터 업데이트
            cursor.execute("""
                UPDATE result 
                SET calib_loc=?, calib_depth=?
                WHERE work_cell=?
            """, (loc, depth, int(cell)))
            conn.commit()
            conn.close()
            self.export_to_excel() # 엑셀 실시간 반영
            print(f"[DB] 셀 {cell} 전체 포인트에 교정 데이터 일괄 저장 완료")
        except Exception as e:
            print(f"[DB ERROR] 교정 데이터 저장 실패: {e}")
        
# =========================================================
# [클래스] 작업 설정 다이얼로그
# =========================================================
class WorkSetupDialog(QDialog):
    def __init__(self, main_window, cell_num):
        super().__init__(main_window)
        self.main = main_window
        self.cell_num = cell_num
        
        self.setWindowTitle(f"셀 {cell_num} 작업 설정")
        self.resize(450, 400)
        self.setStyleSheet("background-color: #2b2b2b; color: white; font-family: 'Arial';")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        lbl_beaker = QLabel(f"셀 {cell_num} 비커(제품) 종류 선택")
        lbl_beaker.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(lbl_beaker)
        
        self.combo_beaker = QComboBox()
        self.combo_beaker.setFixedHeight(40)
        self.combo_beaker.setStyleSheet("background-color: white; color: black; font-size: 14pt;")
        layout.addWidget(self.combo_beaker)
        
        lbl_mode = QLabel("작업 방식 선택")
        lbl_mode.setStyleSheet("font-size: 14pt; font-weight: bold; margin-top: 10px;")
        layout.addWidget(lbl_mode)
        
        mode_layout = QHBoxLayout()
        self.rdo_seq = QRadioButton("순차 작업")
        self.rdo_target = QRadioButton("타겟 작업")
        radio_style = """
            QRadioButton {
                font-size: 16pt; /* 글자 크기 */
            }
            QRadioButton::indicator {
                width: 26px;      /* 동그라미 가로 크기 */
                height: 26px;     /* 동그라미 세로 크기 */
                border-radius: 14px; /* 완벽한 원 생성 */
                border: 2px solid #888888; /* 기본 테두리 회색 */
                background-color: white;   /* 기본 바탕 흰색 */
            }
            QRadioButton::indicator:checked {
                border: 2px solid #0078D7; /* 선택 시 테두리 파란색 */
                /* 중심(cx, cy)부터 안쪽 50%는 파란색, 그 바깥쪽은 흰색으로 칠해서 내부 원을 만듦 */
                background-color: qradialgradient(
                    cx: 0.5, cy: 0.5, radius: 0.5,
                    fx: 0.5, fy: 0.5,
                    stop: 0.0 #0078D7, 
                    stop: 0.5 #0078D7, 
                    stop: 0.6 white, 
                    stop: 1.0 white
                );
            }
        """
        self.rdo_seq.setStyleSheet(radio_style)
        self.rdo_target.setStyleSheet(radio_style)
        self.rdo_seq.setChecked(True)
        mode_layout.addWidget(self.rdo_seq)
        mode_layout.addWidget(self.rdo_target)
        layout.addLayout(mode_layout)
        
        self.stacked_widget = QStackedWidget()
        
        # 1. 순차 작업 설정 위젯
        self.seq_widget = QWidget()
        seq_layout = QGridLayout(self.seq_widget)
        
        lbl_start = QLabel("시작 번호 (1~12):")
        lbl_start.setStyleSheet("font-size: 14pt;")
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, 12)
        self.spin_start.setFixedHeight(40)
        self.spin_start.setStyleSheet("background-color: white; color: black; font-size: 16pt; font-weight: bold;")
        
        lbl_count = QLabel("총 작업 수량:")
        lbl_count.setStyleSheet("font-size: 14pt;")
        self.spin_count = QSpinBox()
        self.spin_count.setRange(0, 12)
        self.spin_count.setFixedHeight(40)
        self.spin_count.setStyleSheet("background-color: white; color: black; font-size: 16pt; font-weight: bold;")
        
        seq_layout.addWidget(lbl_start, 0, 0)
        seq_layout.addWidget(self.spin_start, 0, 1)
        seq_layout.addWidget(lbl_count, 1, 0)
        seq_layout.addWidget(self.spin_count, 1, 1)
        self.stacked_widget.addWidget(self.seq_widget)
        
        # 2. 타겟 작업 설정 위젯
        self.target_widget = QWidget()
        target_layout = QGridLayout(self.target_widget)
        self.chk_points = []
        for i in range(12):
            chk = QCheckBox(f"{i+1}번")
            
            # 테두리와 배경을 명시해서 OS 기본 스타일을 덮어쓰고 크기를 강제 적용
            chk.setStyleSheet("""
                QCheckBox {
                    font-size: 16pt; /* 글자 크기 */
                }
                QCheckBox::indicator {
                    width: 26px;               /* 체크박스 가로 크기 */
                    height: 26px;              /* 체크박스 세로 크기 */
                    border: 2px solid #888888; /* 기본 테두리 회색 */
                    background-color: white;   /* 기본 바탕 흰색 */
                    border-radius: 4px;        /* 모서리를 살짝 둥글게 (디테일) */
                }
                QCheckBox::indicator:checked {
                    border: 2px solid #0078D7; /* 체크 시 테두리 파란색 */
                    background-color: #0078D7; /* 체크 시 네모 안을 파란색으로 꽉 채움 */
                }
            """)
            
            self.chk_points.append(chk)
            target_layout.addWidget(chk, i // 4, i % 4)
        self.stacked_widget.addWidget(self.target_widget)
        
        layout.addWidget(self.stacked_widget)
        layout.addStretch() 
        
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("설정 저장")
        self.btn_cancel = QPushButton("취 소")
        
        self.btn_save.setFixedHeight(50)
        self.btn_cancel.setFixedHeight(50)
        self.btn_save.setStyleSheet("QPushButton { background-color: #0078D7; color: white; font-size: 14pt; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #005A9E; }")
        self.btn_cancel.setStyleSheet("QPushButton { background-color: #555555; color: white; font-size: 14pt; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #777777; }")
        
        self.btn_save.clicked.connect(self.save_settings)
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        self.rdo_seq.toggled.connect(self.on_mode_changed)
        
        self.load_beaker_list()
        self.load_current_values()

        if self.parent():
            parent_geo = self.parent().geometry()
            self.move(parent_geo.x() + int((parent_geo.width() - self.width()) / 2), parent_geo.y() + int((parent_geo.height() - self.height()) / 2))

    def on_mode_changed(self):
        if self.rdo_seq.isChecked():
            self.stacked_widget.setCurrentWidget(self.seq_widget)
        else:
            self.stacked_widget.setCurrentWidget(self.target_widget)

    def load_beaker_list(self):
        try:
            config_path = PRODUCT_PATH / "product.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    beaker_dict = config_data.get("beaker", {})
                    product_list = list(beaker_dict.keys())
                    if product_list:
                        self.combo_beaker.addItems(product_list)
        except Exception:
            pass

    def load_current_values(self):
        cell = self.main.cell_data[self.cell_num]
        work_mode = cell.get("work_mode", "순차 작업")
        target_points = cell.get("target_points", [])
        
        if work_mode == "타겟 작업":
            self.rdo_target.setChecked(True)
            for pt in target_points:
                if 1 <= pt <= 12:
                    self.chk_points[pt-1].setChecked(True)
        else:
            self.rdo_seq.setChecked(True)
            if target_points:
                self.spin_start.setValue(target_points[0])
                self.spin_count.setValue(len(target_points))
            
        current_beaker = self.main.ui.current_beaker.text()
        index = self.combo_beaker.findText(current_beaker)
        if index >= 0:
            self.combo_beaker.setCurrentIndex(index)

    def save_settings(self):
        selected_beaker = self.combo_beaker.currentText()
        target_points = []
        work_mode = "순차 작업"
        
        if self.rdo_seq.isChecked():
            work_mode = "순차 작업"
            start_num = self.spin_start.value()
            count = self.spin_count.value()
            for i in range(count):
                pt = start_num + i
                if pt <= 12:
                    target_points.append(pt)
        else:
            work_mode = "타겟 작업"
            for i, chk in enumerate(self.chk_points):
                if chk.isChecked():
                    target_points.append(i + 1)
                    
        # if not target_points:
        #     self.main.show_message("설정 오류", "작업할 포인트가 선택되지 않았습니다.", QMessageBox.Warning)
        #     return
            
        new_count = len(target_points)
        points_str = ", ".join(map(str, target_points))
        
        msg_text = f"<b>셀 {self.cell_num} ({work_mode})</b><br><br>비커: <b>{selected_beaker}</b><br>작업 포인트: <b>{points_str}</b><br>총 수량: <b>{new_count}개</b><br><br>해당 셀의 기록이 초기화됩니다. 진행하시겠습니까?"
        
        reply = self.main.show_message(
            title="작업 세팅 확인", 
            message=msg_text, 
            icon=QMessageBox.Question,
            buttons=QMessageBox.Ok | QMessageBox.Cancel, 
            font_size=18
        )
        
        if reply == QMessageBox.Ok:
            self.main.ui.current_beaker.setText(selected_beaker)
            self.main.send_product_offsets_to_robot(selected_beaker)
            with self.main.state_lock:
                self.main.cell_data[self.cell_num]["work_mode"] = work_mode
                self.main.cell_data[self.cell_num]["target_points"] = target_points
                self.main.cell_data[self.cell_num]["done_points"] = []
                self.main.cell_data[self.cell_num]["target"] = new_count
                self.main.cell_data[self.cell_num]["pos"] = 0
                self.main.cell_data[self.cell_num]["seq_state"] = "IDLE"
            
            if hasattr(self.main, 'db_manager'):
                try:
                    self.main.db_manager.setup_new_work_by_cell(
                        self.cell_num, new_count, 
                        beaker_type=selected_beaker,
                        work_mode=work_mode,
                        target_points=target_points
                    )
                    self.main.db_manager.update_work_done(str(self.cell_num), 0)
                except Exception as e:
                    print(f"[DB WARN] DB 저장 예외 발생: {e}")
            
            self.main.update_ui_state()
            self.accept()

# =========================================================
# [클래스] UI 팝업 모음
# =========================================================
class CellSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_cell = 0
        self.setWindowTitle("셀 선택")
        self.setFixedSize(400, 250)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint) 
        self.setStyleSheet("background-color: #2b2b2b; color: white; border: 2px solid #0078D7;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        lbl = QLabel("교정 작업을 진행할 셀을 선택하세요.")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 16pt; font-weight: bold; border: none;")
        layout.addWidget(lbl)
        layout.addSpacing(15)
        
        btn_layout = QHBoxLayout()
        btn_cell1 = QPushButton("셀 1 작업")
        btn_cell1.setFixedHeight(60)
        btn_cell1.setStyleSheet("QPushButton { background-color: #0078D7; font-size: 16pt; font-weight: bold; border-radius: 5px; border: none; } QPushButton:hover { background-color: #005A9E; }")
        btn_cell1.clicked.connect(lambda: self.select_cell(1))
        
        btn_cell2 = QPushButton("셀 2 작업")
        btn_cell2.setFixedHeight(60)
        btn_cell2.setStyleSheet("QPushButton { background-color: #28a745; font-size: 16pt; font-weight: bold; border-radius: 5px; border: none; } QPushButton:hover { background-color: #218838; }")
        btn_cell2.clicked.connect(lambda: self.select_cell(2))
        
        btn_layout.addWidget(btn_cell1)
        btn_layout.addWidget(btn_cell2)
        layout.addLayout(btn_layout)
        
        btn_cancel = QPushButton("취 소")
        btn_cancel.setFixedHeight(50)
        btn_cancel.setStyleSheet("QPushButton { background-color: #555555; font-size: 14pt; font-weight: bold; border-radius: 5px; margin-top: 10px; border: none; } QPushButton:hover { background-color: #777777; }")
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_cancel)
        self.setLayout(layout)
        
        if self.parent():
            parent_geo = self.parent().geometry()
            self.move(parent_geo.x() + int((parent_geo.width() - self.width()) / 2), parent_geo.y() + int((parent_geo.height() - self.height()) / 2))

    def select_cell(self, cell_num):
        self.selected_cell = cell_num
        self.accept()

class AlarmDialog(QDialog):
    closed_signal = pyqtSignal() 
    def __init__(self, parent=None, is_error=True, msg=""):
        super().__init__(parent)
        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(500)
        self.setMinimumHeight(300) # 안내 문구를 위해 높이 소폭 증가

        if is_error:
            window_title, title_color, border_color = "로봇 알람 발생", "#FF3333", "#FF3333"
        else:
            window_title, title_color, border_color = "로봇 메시지 알림", "#FFC107", "#FFC107"

        self.setWindowTitle(window_title)
        self.setStyleSheet(f"""
            QDialog {{ background-color: #2b2b2b; border: 2px solid {border_color}; }}
            QLabel {{ background-color: transparent; }}
            QPushButton {{ background-color: #555555; color: #FFFFFF; font-size: 18px; font-weight: bold; border-radius: 5px; border: 1px solid #888; }}
            QPushButton:hover {{ background-color: #0078D7; border: 1px solid #005a9e; }}
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 30, 20, 30)
        
        # ★ 에러 조치 안내 문구 추가
        guide_msg = "<br><br><span style='color:#aaaaaa; font-size:16px;'>※ 에러 원인 제거 후 아래 '확인' 버튼을 눌러주세요.</span>"
        html_message = f'<div style="text-align:left;"><b style="color:{title_color}; font-size:24px;">{window_title}</b><br><br><span style="color:white; font-size:20px;">{msg}</span>{guide_msg}</div>'
        
        lbl_msg = QLabel(html_message)
        lbl_msg.setAlignment(Qt.AlignCenter)
        lbl_msg.setWordWrap(True)
        layout.addWidget(lbl_msg)
        layout.addSpacing(30)

        btn_layout = QHBoxLayout()
        # ★ 버튼명 변경 및 크기 조정
        btn_ok = QPushButton("확인 (닫기)")
        btn_ok.setFixedSize(220, 60)
        btn_ok.clicked.connect(self.close) 
        btn_layout.addStretch(); btn_layout.addWidget(btn_ok); btn_layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        if self.parent():
            parent_geo = self.parent().geometry()
            self.move(parent_geo.x() + int((parent_geo.width() - self.width()) / 2), parent_geo.y() + int((parent_geo.height() - self.height()) / 2))

    def closeEvent(self, event):
        self.closed_signal.emit()
        super().closeEvent(event)

class AlarmThread(QThread):
    alarm_received_signal = pyqtSignal(str, str)
    def __init__(self, main_window, alarm_manager):
        super().__init__()
        self.main = main_window
        self.alarm_manager = alarm_manager
        self.running = True

    def run(self):
        while self.running:
            try:
                current_robot = getattr(self.main, 'robot_30001', None)
                if current_robot is not None and not getattr(self.main, 'robot_disconnected', True):
                    try: alarm = current_robot.alarm_queue.get(timeout=1.0)
                    except queue.Empty: continue 
                        
                    if alarm:
                        self.alarm_manager.process(alarm)
                        if alarm.code is not None:
                            alarm_msg = f"[ALARM {alarm.code}] Sub: {alarm.sub} - Level {alarm.level}"
                            level = "ERROR"
                        else:
                            alarm_msg = f"[MSG] {alarm.msg}"
                            level = "WARN"
                        self.alarm_received_signal.emit(alarm_msg, level)
                else: time.sleep(1)
            except Exception: time.sleep(1)
                
    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        
# -------------------------------------------------------------------------
# ★ [추가/수정] 알람 리셋 통신 및 원격 모드 복귀를 담당하는 스레드
# -------------------------------------------------------------------------
class ResetRobotThread(QThread):
    request_update_signal = pyqtSignal()
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def run(self):
        try:
            # 1. 하드웨어 릴레이 펄스 1초 유지 후 끄기
            time.sleep(1.0)
            self.main.set_variable_with_ui("DO_Alarm_Reset", 0)
            self.main.set_variable_with_ui("initialize", 0)

            # 2. 로봇 전원 상태에 따른 맞춤형 안전 리셋 전송
            if self.main.is_robot_power_on:
                print("[CMD] 로봇 전원 ON 상태: 보호정지 해제 및 다이얼로그 닫기")
                self.main.robot_29999.send_command_29999("unlockProtectiveStop")
                self.main.robot_29999.send_command_29999("closeSafetyDialog")
            else:
                print("[CMD] 로봇 전원 OFF 상태: 세이프티 시스템 전체 리셋 (safety -r)")
                self.main.robot_29999.send_command_29999("safety -r")
                self.main.robot_29999.send_command_29999("unlockProtectiveStop")
                self.main.robot_29999.send_command_29999("closeSafetyDialog")
            
            self.main.robot_29999.send_command_29999("popup -c")
            
            # 3. 로봇이 정상 상태로 돌아오는지 감시 (최대 10초 대기)
            print("[RESET] 알람 리셋 통신 완료. 로봇 정상화(Normal) 대기 중...")
            timeout = 100 # 100 * 0.1초 = 10초
            is_recovered = False
            
            for _ in range(timeout):
                # safety_mode 1 == NORMAL (정상 상태)
                if self.main.safety_mode == 1:
                    is_recovered = True
                    break
                time.sleep(0.1)

            # 4. 정상화 확인 시 원격 제어 모드로 강제 전환
            if is_recovered:
                print("[RESET SUCCESS] 로봇 정상화 확인됨! 원격 모드 전환(robotControl -on) 전송.")
                time.sleep(0.2) # 상태 안정화 찰나 대기
                self.main.robot_29999.send_command_29999("robotControl -on")
            else:
                print("[RESET WARN] 시간 내에 로봇이 정상 상태로 복구되지 않았습니다. 조작기를 확인하세요.")
            
            self.request_update_signal.emit()
            
        except Exception as e:
            print(f"[RESET ERROR] 로봇 복구 시퀀스 중 오류: {e}")

class JobResultDialog(QDialog):
    closed = pyqtSignal(int)
    def __init__(self, parent=None, cell_num=0, point_num=0):
        super().__init__(parent)
        self.cell_num = cell_num
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.resize(320, 160)
        self.setStyleSheet("background-color: #2b2b2b; color: white; border: 2px solid #0078D7;")
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        self.setWindowTitle("작업 완료")
        msg_label = QLabel(f"<b>[셀 {cell_num}] / [포인트 {point_num}]</b> <br>작업이 완료되었습니다.<br>제품을 확인해주세요.")
        msg_label.setAlignment(Qt.AlignCenter)
        msg_label.setStyleSheet("font-size: 14pt; border: none; color: white;")
        layout.addWidget(msg_label)
        layout.addSpacing(10)
        btn_ok = QPushButton("확인 (닫기)")
        btn_ok.setFixedHeight(45)
        btn_ok.setStyleSheet("QPushButton { background-color: #555555; color: white; font-weight: bold; border-radius: 5px; font-size: 12pt; border: 1px solid #888; } QPushButton:hover { background-color: #0078D7; border: 1px solid #005a9e; }")
        btn_ok.clicked.connect(self.close)
        layout.addWidget(btn_ok)
        self.setLayout(layout)
        if self.parent():
            parent_geo = self.parent().geometry()
            self.move(parent_geo.x() + int((parent_geo.width() - self.width()) / 2), parent_geo.y() + int((parent_geo.height() - self.height()) / 2))

    def closeEvent(self, event):
        self.closed.emit(self.cell_num)
        super().closeEvent(event)

class ConnectionThread(QThread):
    finished_signal = pyqtSignal(bool, str)
    def __init__(self, robot_29999, robot_30001, modbus_client):
        super().__init__()
        self.robot_29999 = robot_29999
        self.robot_30001 = robot_30001
        self.modbus_client = modbus_client

    def run(self):
        try:
            sock1 = self.robot_29999.connect_29999()
            sock2 = self.robot_30001.connect_30001()
            modbus_ok = False
            if self.modbus_client is not None:
                modbus_ok = self.modbus_client.connect()

            if sock1 is None: self.finished_signal.emit(False, "29999 포트 연결 실패"); return
            if sock2 is None: self.finished_signal.emit(False, "30001 포트 연결 실패"); return
            if not modbus_ok: self.finished_signal.emit(False, "Modbus(502) 연결 실패"); return
            
            self.finished_signal.emit(True, "로봇 시스템 연결 성공")
        except Exception as e:
            self.finished_signal.emit(False, f"연결 중 예외 발생: {str(e)}")

# =========================================================================
# [클래스] 메인 어플리케이션 GODO
# =========================================================================
class GODO(QMainWindow):
    log_signal = pyqtSignal(str)
    report_alarm_signal = pyqtSignal(str, str)
    update_var_signal = pyqtSignal(str, object)
    
    def __init__(self):
        super(GODO, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.sys_out_stream = EmittingStream()
        self.sys_out_stream.textWritten.connect(self.append_system_msg)
        sys.stdout = self.sys_out_stream # print() 출력을 가로챔
        sys.stderr = self.sys_out_stream # (선택) 에러 로그도 보고싶다면 주석 해제
        
        if hasattr(self.ui, 'godo_msg_area'):
            self.ui.godo_msg_area.setReadOnly(True)
        if hasattr(self.ui, 'system_msg_area'):
            self.ui.system_msg_area.setReadOnly(True)
            
        self.update_log_paths()
        
        self.cleanup_old_images()
        
        self.clock_timer = QtCore.QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        
        if hasattr(self.ui, 'quality_log_btn'):
            self.ui.quality_log_btn.clicked.connect(self.on_quality_log_btn_clicked)

        self.logo_path = LOGO_PATH
        if os.path.exists(self.logo_path):
            self.ui.logo.setPixmap(QtGui.QPixmap(str(self.logo_path)))
            
        # ==========================================
        # ★ 비전 매니저, DB, 시퀀스 상태 통합 관리
        # ==========================================
        self.db_manager = DBManager()
        self.db_manager.init_result_table()

        # ==========================================
        # ★ 비전 매니저 중간 다리(Bridge) 스레드 구동
        # ==========================================
        self.is_vision_ready = False
        try:
            vision_cfg_path = VISION_PATH / "vision_config.json"
            product_cfg_path = PRODUCT_PATH / "product.json"  # ★ 추가
            # ★ 수정: product 경로도 같이 넘겨줌
            self.vision_bridge = VisionBridgeQThread(vision_cfg_path, product_cfg_path)
            self.vision_bridge.image_signal.connect(self.update_camera_image)
            self.vision_bridge.result_signal.connect(self.on_vision_result_ready)
            self.vision_bridge.ready_signal.connect(self.on_vision_worker_ready)
            self.vision_bridge.start()
        except Exception as e:
            print(f"[DEBUG] 비전 브릿지 스레드 구동 실패: {e}")
            
        self.state_lock = threading.Lock()
        
        goals = self.db_manager.get_work_goals()
        done_counts = self.db_manager.get_work_done_counts()
        saved_beaker = goals.get('1', {}).get('beaker', "")
        if saved_beaker: self.ui.current_beaker.setText(saved_beaker)

        # # ★ cell_data 통합 (DB 로드 연동 완료)
        # self.cell_data = {
        #     1: {"target": goals.get('1', {}).get('count', 0), 
        #         "pos": done_counts.get('1', 0), 
        #         "work_mode": goals.get('1', {}).get('work_mode', "순차 작업"), 
        #         "target_points": goals.get('1', {}).get('target_points', []), 
        #         "done_points": goals.get('1', {}).get('done_points', []),
        #         "last_action": "none", "sample_id": "", "seq_state": "IDLE", "req_off_time": 0, "sensor_off_time": 0},
        #     2: {"target": goals.get('2', {}).get('count', 0), 
        #         "pos": done_counts.get('2', 0), 
        #         "work_mode": goals.get('2', {}).get('work_mode', "순차 작업"), 
        #         "target_points": goals.get('2', {}).get('target_points', []), 
        #         "done_points": goals.get('2', {}).get('done_points', []),
        #         "last_action": "none", "sample_id": "", "seq_state": "IDLE", "req_off_time": 0, "sensor_off_time": 0}
        # }
        # ★ cell_data 통합 (DB 로드 연동 완료)
        self.cell_data = {
            1: {"target": goals.get('1', {}).get('count', 0), 
                "pos": done_counts.get('1', 0), 
                "work_mode": goals.get('1', {}).get('work_mode', "순차 작업"), 
                "target_points": goals.get('1', {}).get('target_points', []), 
                "done_points": goals.get('1', {}).get('done_points', []),
                "last_action": "none", "sample_id": "", "seq_state": "IDLE", "req_off_time": 0, "sensor_off_time": 0,
                "calib_loc": "", "beaker_loc": "", "loc_dev": "", "calib_depth": "", "beaker_depth": "", "depth_dev": ""},
            2: {"target": goals.get('2', {}).get('count', 0), 
                "pos": done_counts.get('2', 0), 
                "work_mode": goals.get('2', {}).get('work_mode', "순차 작업"), 
                "target_points": goals.get('2', {}).get('target_points', []), 
                "done_points": goals.get('2', {}).get('done_points', []),
                "last_action": "none", "sample_id": "", "seq_state": "IDLE", "req_off_time": 0, "sensor_off_time": 0,
                "calib_loc": "", "beaker_loc": "", "loc_dev": "", "calib_depth": "", "beaker_depth": "", "depth_dev": ""}
        }
        
        # # ====================================================================
        # # ★ [수정] 프로그램 시작 시 JSON 설정 파일에서 영구 보존된 교정값을 1순위로 불러옵니다!
        # # ====================================================================
        # self.calib_reference = {1: None, 2: None} 
        # try:
        #     v_cfg_path = VISION_PATH / "vision_config.json"
        #     if v_cfg_path.exists():
        #         with open(v_cfg_path, 'r', encoding='utf-8') as f:
        #             v_data = json.load(f)
        #             calib_saved = v_data.get("CALIB_DATA", {})
        #             for area_str in ["1", "2"]:
        #                 saved_val = calib_saved.get(area_str)
        #                 if saved_val:
        #                     # 저장된 딕셔너리를 튜플 (cx, cy, depth) 형태로 복원
        #                     self.calib_reference[int(area_str)] = (saved_val["cx"], saved_val["cy"], saved_val["depth"])
        #                     print(f"[SYSTEM] 셀 {area_str} 이전 교정값 로드 완료: {self.calib_reference[int(area_str)]}")
        # except Exception as e:
        #     print(f"[SYSTEM WARN] 이전 교정값 로드 실패: {e}")
        
        self.active_popups = {}
        
        self.robot_config_file = os.path.join(ROBOT_PATH, "robot_config.json")
        self.robot_config = self.load_robot_config()
        
        self.robot_ip = self.robot_config["ROBOT_IP"]
        self.robot_port1 = self.robot_config["ROBOT_PORT_1"]
        self.robot_port2 = self.robot_config["ROBOT_PORT_2"]
        self.modbus_port = self.robot_config["ROBOT_PORT_3"]
        
        self.cached_robot_vars = {}
        self.modbus_lock = threading.Lock()
        self.create_robot_objects()
        
        # self.update_ui_state()
        self.robot_disconnected = True
        self.init_robot_var_ui()
        
        self.alarm_manager = AlarmManager()
        self.alarm_thread = None
        self.alarm_popup_shown = False
        self.active_alarm_dialog = None
        
        self.is_working = False
        
        if hasattr(self.ui, 'start_btn'):
            self.ui.start_btn.setEnabled(True)
            self.ui.stop_btn.setEnabled(False)
            self.ui.mode_circle.setStyleSheet("border-radius: 30px; background-color: #28a745;")
            self.ui.mode_label.setText("수동 모드")

        self._speed_initialized = False
        self.ui.speed_slider.valueChanged.connect(self.on_slider_changed)
        self.on_slider_changed(self.ui.speed_slider.value())
        self.ui.speed_slider.valueChanged.connect(self.on_speed_changed)

        self.ui.stop_btn.clicked.connect(self.on_stop_button_clicked)
        self.ui.main_btn.clicked.connect(self.on_main_btn_clicked)
        self.ui.manage_btn.clicked.connect(self.on_manage_btn_clicked)
        self.ui.robot_btn.clicked.connect(self.on_robot_btn_clicked)
        self.ui.work_reset_btn.clicked.connect(self.reset_workload)
        if hasattr(self.ui, 'done_work_reset_btn'):
            self.ui.done_work_reset_btn.clicked.connect(self.reset_done_workload)
        self.ui.start_btn.clicked.connect(self.on_start_work_btn_clicked)
        self.ui.robot_power_on_btn.clicked.connect(lambda: self.on_robot_power_on_button_clicked(True))
        self.ui.alarm_reset_btn.clicked.connect(self.robot_alarm_reset_button)
        
        # # 새로 추가된 버튼 연결 (UI 파일에 해당 객체가 있다고 가정)
        # if hasattr(self.ui, 'buzzer_mute_btn'):
        #     self.ui.buzzer_mute_btn.clicked.connect(self.on_buzzer_mute_clicked)
        # if hasattr(self.ui, 'alarm_rst_btn'):
        #     self.ui.alarm_rst_btn.clicked.connect(self.on_alarm_rst_btn_clicked)
        
        self.is_inspection_mode_cached = True  # UI 상태 캐시용
        self.current_beaker_name_cached = ""   # 비커 이름 캐시용
            
        self.is_buzzer_muted = False
        
        self.ui.calibration_mode_btn.setCheckable(True)
        self.ui.inspection_mode_btn.setCheckable(True)
        self.ui.calibration_mode_btn.toggled.connect(self.on_calibration_mode_toggled)
        self.ui.inspection_mode_btn.toggled.connect(self.on_inspection_mode_toggled)
        self.ui.inspection_mode_btn.setChecked(True)
        
        self.ui.cell_1_work_goal_label.installEventFilter(self)
        self.ui.cell_2_work_goal_label.installEventFilter(self)
        
        if hasattr(self.ui, 'label_cam'):
            self.ui.label_cam.installEventFilter(self)
        
        toggle_buttons = [
            self.ui.home_pose_btn, self.ui.cell_1_pose_btn, self.ui.cell_2_pose_btn,
            self.ui.jig1_point_btn_1, self.ui.jig1_point_btn_2, self.ui.jig1_point_btn_3, self.ui.jig1_point_btn_4, 
            self.ui.jig1_point_btn_5, self.ui.jig1_point_btn_6, self.ui.jig1_point_btn_7, self.ui.jig1_point_btn_8,
            self.ui.jig1_point_btn_9, self.ui.jig1_point_btn_10, self.ui.jig1_point_btn_11, self.ui.jig1_point_btn_12,
            self.ui.jig2_point_btn_1, self.ui.jig2_point_btn_2, self.ui.jig2_point_btn_3, self.ui.jig2_point_btn_4, 
            self.ui.jig2_point_btn_5, self.ui.jig2_point_btn_6, self.ui.jig2_point_btn_7, self.ui.jig2_point_btn_8,
            self.ui.jig2_point_btn_9, self.ui.jig2_point_btn_10, self.ui.jig2_point_btn_11, self.ui.jig2_point_btn_12,
        ]
        for btn in toggle_buttons:
            btn.setCheckable(True)
            
        self.init_manual_control_buttons()
        
        self.auto_mode = False
        self.manual_mode = False
        self.calibration_mode = False
        self.inspection_mode = False
        
        self.init_state = "IDLE"  # "IDLE" -> "REQ_INIT" -> "WAIT_INIT_TRUE" -> "WAIT_INIT_FALSE" -> "READY"
        self.was_remote = False    # 이전 원격 모드 상태 저장용
        
        self.is_robot_power_on = False
        self.is_task_running = False
        self.is_task_paused = False
        self.safety_mode = 0
        self.robot_mode = 0
        self.robot_speed = None
        
        self.values = (self.is_robot_power_on, self.is_task_running, self.is_task_paused, self.safety_mode, self.robot_mode, self.robot_speed)
        self.actual_joint_base = 0.0; self.actual_joint_shoulder = 0.0; self.actual_joint_elbow = 0.0
        self.actual_joint_wrist1 = 0.0; self.actual_joint_wrist2 = 0.0; self.actual_joint_wrist3 = 0.0
        
        self.pause_event = threading.Event()
        self.pause_event.set()
        
        self.report_alarm_signal.connect(self.add_alarm_log)
        self.update_var_signal.connect(self._do_update_robot_var_ui)
        
        self.state_timer = QtCore.QTimer(self)
        self.state_timer.timeout.connect(self.poll_robot_state)
        self.state_timer.start(100)
        
        QtCore.QTimer.singleShot(100, self.start_background_connection)
        
        self.update_ui_state()
        
        # 외부 통신 TCP 서버 구동 (포트 5000)
        self.external_trigger_thread = ExternalTcpServerThread(self, port=5000)
        self.external_trigger_thread.start(QThread.HighestPriority)
        # self.external_trigger_thread.start()
        
        # ★ 2. 외부 통신 TCP 서버 구동 아래에 추가
        self.remote_status_timer = QtCore.QTimer(self)
        self.remote_status_timer.timeout.connect(self.poll_remote_status)
        self.remote_status_timer.start(1500) # 1.5초마다 29999포트 감시
        
    @pyqtSlot(bool)
    def on_vision_worker_ready(self, is_ready):
        self.is_vision_ready = is_ready
        if is_ready:
            print("[DEBUG] 비전 백그라운드 상주 스레드 연결 성공 (통신 간섭 없음)")
        else:
            print("[DEBUG] 비전 카메라 연결 실패")
            # ★ 카메라 연결 실패 시 화면 중앙에 강력한 경고 팝업을 띄웁니다.
            self.show_message(
                title="카메라 연결 실패",
                message="<b>[카메라 오류]</b><br>비전 카메라 네트워크 연결에 실패했습니다.<br><br>카메라 전원과 랜선(IP)을 확인한 후<br><b>프로그램을 종료하고 다시 실행해주세요.</b>",
                icon=QMessageBox.Critical,
                buttons=QMessageBox.Ok,
                button_color="#dc3545"  # 빨간색 경고 버튼
            )
                    
    def send_product_offsets_to_robot(self, beaker_name=None):
        if getattr(self, 'robot_disconnected', True): return
        
        if not beaker_name: beaker_name = self.ui.current_beaker.text()
        if not beaker_name: return

        try:
            config_path = PRODUCT_PATH / "product.json"
            if config_path.exists():
                # 1. 파일 읽기
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                beaker_data = data.get("beaker", {}).get(beaker_name)
                if not beaker_data: return

                # ★ 1. 삼각함수로 jig_center_offs 자동 계산
                diameter = beaker_data.get("target_mm", beaker_data.get("diameter", 0))
                radius = diameter / 2.0
                v_angle_deg = 114.53 
                
                if radius > 0:
                    half_angle_rad = math.radians(v_angle_deg / 2.0)
                    calc_center_dist = radius / math.sin(half_angle_rad)
                    jig_center = int(round(calc_center_dist))
                else:
                    jig_center = 0

                # =========================================================
                # ★ [추가] 계산된 jig_center 값을 JSON 데이터에 갱신하고 파일에 저장!
                # =========================================================
                if beaker_data.get("jig_center_offs") != jig_center:
                    beaker_data["jig_center_offs"] = jig_center
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    print(f"[INFO] '{beaker_name}' jig_center_offs({jig_center}) product.json 파일에 저장 완료.")

                # ★ 2. 나머지는 사용자가 JSON에 입력한 값을 그대로 사용
                jig_height = int(beaker_data.get("jig_height_offs", 0))
                cell_deep = int(beaker_data.get("cell_deep_offs", 0))
                sensing = int(beaker_data.get("sensing_offs", 0)) 

                self.set_variable_with_ui("jig_center_offs", jig_center)
                self.set_variable_with_ui("jig_height_offs", jig_height)
                self.set_variable_with_ui("cell_deep_offs", cell_deep)
                self.set_variable_with_ui("sensing_offs", sensing)
                
                print(f"[INFO] '{beaker_name}' 오프셋 전송 완료: Center(계산)={jig_center}, Height(입력)={jig_height}, Deep(입력)={cell_deep}, Sensing(입력)={sensing}")
        except Exception as e:
            print(f"[ERROR] 비커 오프셋 계산 및 전송 실패: {e}")
            
    def safe_append_log(self, ui_area, html_msg):
        """스마트 자동 스크롤: 맨 아래일 때만 최신 내용으로 스크롤하고, 위를 볼 땐 고정합니다."""
        if not ui_area: return
        
        # 현재 스크롤바 위치 확인
        scrollbar = ui_area.verticalScrollBar()
        # 하단에서 약 20px 이내에 있으면 '맨 아래'로 간주 (버퍼 부여)
        at_bottom = scrollbar.value() >= (scrollbar.maximum() - 20)
        
        # 텍스트 추가
        ui_area.append(html_msg)
        
        # 맨 아래에 있었다면 추가 후에도 맨 아래로 이동
        if at_bottom:
            ui_area.moveCursor(QtGui.QTextCursor.End)
            
    def cleanup_old_images(self):
        """vision_config.json의 SAVE_CYCLE을 기준으로 오래된 이미지를 삭제합니다."""
        try:
            v_cfg_path = VISION_PATH / "vision_config.json"
            save_cycle = "1_month" # 기본값
            if v_cfg_path.exists():
                with open(v_cfg_path, 'r', encoding='utf-8') as f:
                    v_data = json.load(f)
                    save_cycle = v_data.get("SAVE_CYCLE", "1_month")
            
            parts = save_cycle.split("_")
            if len(parts) != 2: return
            num = int(parts[0])
            unit = parts[1]
            
            days_to_keep = 30
            if unit == "week": days_to_keep = num * 7
            elif unit == "month": days_to_keep = num * 30
            else: return
            
            # 기준일 계산 (오늘 날짜 - 보관 일수)
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            img_dir = DB_PATH / "Log" / "Image"
            
            if img_dir.exists():
                deleted_count = 0
                for filepath in img_dir.glob("*.jpg"):
                    # 파일명 형식: YYYYMMDD_HHMMSS_...
                    parts = filepath.stem.split("_")
                    if len(parts) >= 2:
                        date_str = parts[0]
                        if len(date_str) == 8 and date_str.isdigit():
                            file_date = datetime.datetime.strptime(date_str, "%Y%m%d")
                            if file_date < cutoff_date:
                                filepath.unlink()
                                deleted_count += 1
                if deleted_count > 0:
                    print(f"[SYSTEM] 보관 주기({save_cycle}) 경과된 예전 비전 이미지 {deleted_count}장 삭제 완료.")
        except Exception as e:
            print(f"[SYSTEM ERROR] 예전 이미지 삭제 중 오류: {e}")
            
    @pyqtSlot(str)
    def append_godo_msg_safe(self, html_msg):
        """TCP 백그라운드 스레드 대신 메인 스레드가 안전하게 UI 창에 글씨를 써줍니다."""
        self.safe_append_log(self.ui.godo_msg_area, html_msg)
            
    # -------------------------------------------------------------------
    # ★ 실시간 시계 및 TXT 로그 관리 헬퍼 함수
    # -------------------------------------------------------------------
    def update_clock(self):
        if hasattr(self.ui, 'dateEdit'):
            now = QDateTime.currentDateTime()
            try:
                # dateEdit가 QDateTimeEdit 위젯일 경우
                self.ui.dateEdit.setDateTime(now) 
            except AttributeError:
                try:
                    # dateEdit가 QLineEdit 또는 QLabel 위젯일 경우
                    self.ui.dateEdit.setText(now.toString("yyyy-MM-dd HH:mm:ss"))
                except: pass

    def update_log_paths(self, is_reset=False):
        now = QDateTime.currentDateTime()
        month_str = now.toString("yyyy-MM")     # 예: 2026-04 (월별 폴더용)
        day_str = now.toString("yyyy-MM-dd")    # 예: 2026-04-16 (일별 폴더용)
        today = now.toString("yyyyMMdd")        # 예: 20260416 (파일명 접두사)
        
        # ★ [수정] 경로 세분화: DB/Log/GODO/2026-04/2026-04-16/
        log_dir = DB_PATH / "Log"
        godo_dir = log_dir / "GODO" / month_str / day_str
        sys_dir = log_dir / "System" / month_str / day_str
        
        godo_dir.mkdir(parents=True, exist_ok=True)
        sys_dir.mkdir(parents=True, exist_ok=True)
        
        if not hasattr(self, 'log_date') or self.log_date != today:
            self.log_date = today
            max_idx = 0
            # 해당 날짜 폴더 안에서 가장 높은 번호를 스캔
            for d in [godo_dir, sys_dir]:
                for f in d.glob(f"{today}_*.txt"):
                    try:
                        idx = int(f.stem.split("_")[1])
                        if idx > max_idx: max_idx = idx
                    except: pass
            self.log_index = max_idx + 1
        elif is_reset:
            self.log_index += 1
            
        self.godo_log_path = godo_dir / f"{self.log_date}_{self.log_index}.txt"
        self.sys_log_path = sys_dir / f"{self.log_date}_{self.log_index}.txt"
        
    def prepend_log_to_txt(self, path, new_line):
        try:
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                
            with open(path, 'a', encoding='utf-8') as f:
                f.write(new_line + "\n")
        except Exception as e:
            print(f"[TXT LOG ERROR] {e}")
            
    def append_log_to_txt(self, path, new_line):
        try:
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                
            with open(path, 'a', encoding='utf-8') as f:
                f.write(new_line + "\n")
        except Exception as e:
            print(f"[TXT LOG ERROR] {e}")

    def on_quality_log_btn_clicked(self):
        log_dir = DB_PATH / "Log"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 탐색기 화면 열기
        file_path, _ = QFileDialog.getOpenFileName(self, "로그 파일 열기", str(log_dir), "텍스트 파일 (*.txt);;모든 파일 (*)")
        if file_path:
            try:
                if platform.system() == 'Windows':
                    os.startfile(file_path) # 윈도우 메모장으로 바로 열기
                elif platform.system() == 'Darwin':
                    subprocess.call(['open', file_path])
                else:
                    subprocess.call(['xdg-open', file_path])
            except Exception as e:
                self.show_message("파일 열기 실패", f"파일을 열 수 없습니다:\n{e}", QMessageBox.Warning)

    # =========================================================================
    # [TCP 처리 엔진] 외부 장비(GODO)의 JSON 요청 분석 및 상태 연동
    # =========================================================================
    def process_external_request(self, req):
        action = req.get("action", "").lower() 
        area = int(req.get("area", 1))
        
        is_insp_mode = self.is_inspection_mode_cached
        beaker_name = self.current_beaker_name_cached
        
        if area not in [1, 2]:
            return {"area": area, "result": "fail", "result_code": 2, "msg": "잘못된 트레이 구역(area) 번호입니다."}

        with self.state_lock:
            cell = self.cell_data[area]

            if action == "req_state":
                is_startup_busy = getattr(self, 'is_working', False) and not getattr(self, 'is_startup_ready', False)
                is_robot_at_home = self.cached_robot_vars.get("is_home", 0) == 1
                
                active_states = ["WAIT_INPUT_START", "INPUT_ING", "VISION_MEASURING", "WAIT_OUTPUT_START", "OUTPUT_ING", "WAIT_IS_HOME", "CALIB_ING", "WAIT_CALIB_EXIT"]
                is_moving_state = any(self.cell_data[c]["seq_state"] in active_states for c in [1, 2])

                is_hardware_error = self.is_alarm_state() or getattr(self, 'cyl1_alarm_triggered', False) or getattr(self, 'cyl2_alarm_triggered', False)

                if self.robot_disconnected:
                    state_str, msg = "error", "로봇 연결 끊김"
                elif is_hardware_error: 
                    state_str, msg = "error", "로봇 에러(알람) 발생 상태"
                elif not getattr(self, 'auto_mode', False):
                    state_str, msg = "error", "로봇 수동 모드"
                elif not is_insp_mode:
                    state_str, msg = "error", "시료 비커 모드 아님"
                elif is_startup_busy:
                    state_str, msg = "busy", "로봇 초기화 중"
                elif is_moving_state or not is_robot_at_home:
                    state_str, msg = "busy", "로봇 동작 중"
                else:
                    state_str, msg = "waiting", "" 
                
                curr_pt = cell.get("current_point", 0)
                last_pos_val = curr_pt if curr_pt > 0 else -1
                
                total_target_count = cell.get("target", 0)
                
                return {
                    "area": area, "state": state_str, "last_action": cell["last_action"], "last_pos": last_pos_val,
                    "total_target": total_target_count, 
                    "sample_id": cell.get("sample_id", ""),
                    "calib_loc": cell.get("calib_loc", ""), "beaker_loc": cell.get("beaker_loc", ""),
                    "loc_deviation": cell.get("loc_dev", ""), "calib_depth": cell.get("calib_depth", ""),
                    "beaker_depth": cell.get("beaker_depth", ""), "depth_deviation": cell.get("depth_dev", ""),
                    "msg": msg
                }
                
            # =================================================================
            # ★ 최우선 방어막 (연결 끊김 및 하드웨어 알람)
            # =================================================================
            if self.robot_disconnected:
                return {"area": area, "result": "fail", "result_code": 2, "msg": "로봇 연결 끊김"}

            is_hardware_error = self.is_alarm_state() or getattr(self, 'cyl1_alarm_triggered', False) or getattr(self, 'cyl2_alarm_triggered', False)
            if is_hardware_error:
                return {"area": area, "result": "fail", "result_code": 2, "msg": "로봇 에러(알람) 발생 상태"}
            
            # =================================================================
            # ★ 개별 액션 처리
            # =================================================================
            if action == "input":
                # -------------------------------------------------------------
                # 1순위 검사: '시료 유무' (작업 정지 상태와 무관하게 가장 먼저 확인!)
                # -------------------------------------------------------------
                is_empty = False
                work_mode = cell.get("work_mode", "순차 작업")
                
                if cell["target"] == 0:
                    is_empty = True
                elif work_mode == "순차 작업" and cell.get("pos", 0) >= cell["target"]:
                    is_empty = True
                elif work_mode == "타겟 작업" and cell.get("target_points") and len(cell.get("done_points", [])) >= len(cell["target_points"]):
                    is_empty = True

                if is_empty:
                    # 텅 빈 것이 확인되면 전체 시스템 자동 종료 체크를 한 번 더 수행
                    self.check_and_auto_stop()
                    # 정지 여부 상관없이 묻지도 따지지도 않고 '시료 없음' 리턴!
                    return {"area": area, "result": "fail", "result_code": 1, "msg": "투입 할 시료가 없습니다."}

                # -------------------------------------------------------------
                # 2순위 검사: 시료가 '남아있는데' 프로그램이 정지(Stop) 상태라면 차단
                # -------------------------------------------------------------
                if not getattr(self, 'is_working', False):
                    print(f"[TCP 차단] 작업 시작 버튼이 눌리지 않아 {action} 명령 거부")
                    return {"area": area, "result": "fail", "result_code": 2, "msg": "PC 프로그램의 '작업 시작' 버튼이 눌리지 않았습니다."}

                # -------------------------------------------------------------
                # 3순위: 기타 로봇/프로그램 조건
                # -------------------------------------------------------------
                if not getattr(self, 'auto_mode', False) or not is_insp_mode:
                    return {"area": area, "result": "fail", "result_code": 2, "msg": "수동모드이거나 검사 모드가 아닙니다."}
                
                if cell["last_action"] == "input":
                    return {"area": area, "result": "fail", "result_code": 2, "msg": "작업요청 순서 오류 (Output 필요)"}
                
                active_states = ["WAIT_INPUT_START", "INPUT_ING", "VISION_MEASURING", "WAIT_OUTPUT_START", "OUTPUT_ING", "WAIT_IS_HOME", "CALIB_ING", "WAIT_CALIB_EXIT"]
                if self.cell_data[1]["seq_state"] in active_states or self.cell_data[2]["seq_state"] in active_states:
                    return {"area": area, "result": "fail", "result_code": 2, "msg": "로봇이 현재 다른 작업을 수행 중입니다."}
                
                # 통과 -> 포인트 계산 후 예약/진행
                if work_mode == "타겟 작업" and cell.get("target_points"): 
                    next_pos = cell["target_points"][len(cell.get("done_points", []))]
                else: 
                    next_pos = cell["pos"] + 1

                if not getattr(self, 'is_startup_ready', False):
                    cell["current_point"] = next_pos
                    cell["seq_state"] = "PENDING_INPUT"
                    print(f"[TCP] 초기화 중 셀 {area} INPUT 예약 접수 (포인트 {next_pos})")
                    return {"area": area, "result": "ok", "result_code": 0, "msg": "로봇 초기화가 완료되면 즉시 시작됩니다."}

                cell["current_point"] = next_pos 
                self.set_variable_with_ui(f"J{area}_target_point", next_pos)
                self.set_variable_with_ui(f"c{area}_input_job_req", 1)

                cell["last_action"] = "input"
                cell["sample_id"], cell["calib_loc"], cell["beaker_loc"], cell["loc_dev"], cell["calib_depth"], cell["beaker_depth"], cell["depth_dev"] = "", "", "", "", "", "", ""
                cell["seq_state"] = "WAIT_INPUT_START" 
                
                if hasattr(self, 'db_manager'):
                    self.db_manager.insert_new_job(area, next_pos, beaker_name)
                QtCore.QMetaObject.invokeMethod(self, "update_ui_state", Qt.QueuedConnection)
                return {"area": area, "result": "ok", "result_code": 0, "msg": ""}

            elif action == "output":
                # Output은 '빈 상태'라는 개념이 없으므로 정지 상태면 바로 차단
                if not getattr(self, 'is_working', False):
                    print(f"[TCP 차단] 작업 시작 버튼이 눌리지 않아 {action} 명령 거부")
                    return {"area": area, "result": "fail", "result_code": 2, "msg": "PC 프로그램의 '작업 시작' 버튼이 눌리지 않았습니다."}

                if not getattr(self, 'auto_mode', False) or not self.ui.inspection_mode_btn.isChecked():
                    return {"area": area, "result": "fail", "result_code": 2, "msg": "수동모드이거나 검사 모드가 아닙니다."}
                
                if cell["last_action"] in ["none", "output"]:
                    return {"area": area, "result": "fail", "result_code": 2, "msg": "작업요청 순서 오류 (Input 필요)"}

                active_states = ["WAIT_INPUT_START", "INPUT_ING", "VISION_MEASURING", "WAIT_OUTPUT_START", "OUTPUT_ING", "WAIT_IS_HOME", "CALIB_ING", "WAIT_CALIB_EXIT"]
                if self.cell_data[1]["seq_state"] in active_states or self.cell_data[2]["seq_state"] in active_states:
                    return {"area": area, "result": "fail", "result_code": 2, "msg": "로봇이 현재 다른 작업을 수행 중입니다."}

                curr_pt = cell.get("current_point", cell["pos"])

                if not getattr(self, 'is_startup_ready', False):
                    cell["current_point"] = curr_pt
                    cell["seq_state"] = "PENDING_OUTPUT"
                    print(f"[TCP] 초기화 중 셀 {area} OUTPUT 예약 접수 (포인트 {curr_pt})")
                    return {"area": area, "result": "ok", "result_code": 0, "msg": "로봇 초기화가 완료되면 즉시 시작됩니다."}

                self.set_variable_with_ui(f"J{area}_target_point", curr_pt)
                self.set_variable_with_ui(f"c{area}_output_job_req", 1)

                cell["last_action"] = "output"
                cell["seq_state"] = "WAIT_OUTPUT_START"
                QtCore.QMetaObject.invokeMethod(self, "update_ui_state", Qt.QueuedConnection)
                return {"area": area, "result": "ok", "result_code": 0, "msg": ""}

            elif action == "reset":
                self.reset_flag_all = True
                return {"area": area, "result": "ok", "result_code": 0, "msg": "전체 셀 초기화 명령 접수 완료"}
            
        return {"area": area, "result": "fail", "result_code": 2, "msg": "알 수 없는 action 입니다."}
    
    def poll_remote_status(self):
        """29999포트에 remoteControl -status를 전송하여 원격/로컬 상태 감시"""
        if getattr(self, '_is_polling_remote', False): 
            return
            
        self._is_polling_remote = True

        def _check():
            try:
                # 로봇 객체가 생성되어 있는지 안전하게 확인
                if not hasattr(self, 'robot_29999') or self.robot_29999 is None:
                    return

                # 대시보드 서버는 로컬 모드일 때도 열려있으므로 상태를 얻어올 수 있음
                res = self.robot_29999.send_command_29999("remoteControl -status")
                
                if res:
                    res_str = res.lower()
                    # ====================================================================
                    # ★ [핵심 수정] 가비지 데이터(다른 스레드의 응답) 차단 필터링
                    # 오직 명확하게 true/enabled 또는 false/disabled 단어가 있을 때만 상태를 바꿉니다.
                    # ====================================================================
                    if "false" in res_str or "disabled" in res_str:
                        QtCore.QMetaObject.invokeMethod(self, "handle_remote_status", Qt.QueuedConnection, QtCore.Q_ARG(bool, False))
                    elif "true" in res_str or "enabled" in res_str:
                        QtCore.QMetaObject.invokeMethod(self, "handle_remote_status", Qt.QueuedConnection, QtCore.Q_ARG(bool, True))
                    else:
                        # "RobotMode: 5" 같이 다른 명령의 응답이 섞여 들어오면 그냥 쿨하게 무시합니다! (깜빡임 완벽 차단)
                        pass
            except: 
                pass
            finally:
                self._is_polling_remote = False

        threading.Thread(target=_check, daemon=True).start()

    @pyqtSlot(bool)
    def handle_remote_status(self, is_remote):
        # ★ 1. 원격(Remote) / 로컬(Local) 상태는 PC의 통신 제어 권한을 의미합니다. (자동/수동과 별개)
        if getattr(self, 'is_remote_mode', False) != is_remote:
            self.is_remote_mode = is_remote
            if not is_remote:
                print("[SYSTEM] 로봇 로컬(Local) 제어 권한 감지 -> PC 제어 불가 (연결 끊김 처리)")
                self.on_robot_disconnected()
            else:
                print("[SYSTEM] 로봇 원격(Remote) 제어 권한 감지 -> 자동 재연결 시도")
                if not getattr(self, '_is_reconnecting', False):
                    self._is_reconnecting = True
                    self.connect_robot(show_popup=False)
                    QtCore.QTimer.singleShot(3000, lambda: setattr(self, '_is_reconnecting', False))
                    
        # 2. 프로그램은 '연결 끊김'인데, 로봇은 '원격 모드'일 때 강제 복구
        elif is_remote and getattr(self, 'robot_disconnected', True):
            if not getattr(self, '_is_reconnecting', False):
                print("[SYSTEM] 연결 끊김 상태에서 원격 모드 복귀 확인 -> 강제 재연결 시도")
                self._is_reconnecting = True
                self.connect_robot(show_popup=False)
                QtCore.QTimer.singleShot(3000, lambda: setattr(self, '_is_reconnecting', False))

    def poll_robot_state(self):
        if getattr(self, '_is_polling', False): return
        self._is_polling = True
        
        try:
            if hasattr(self, 'conn_thread') and self.conn_thread.isRunning():
                return

            if getattr(self, 'reset_flag_all', False):
                self.reset_flag_all = False
                self.headless_reset_all()
                
            # 이제 안전하게 데이터를 가져옵니다.
            data = self.robot_30001.get_data()
            
            if self.robot_disconnected:
                self.robot_status_update()
            
            if hasattr(self, 'modbus_client') and self.modbus_client is not None:
                try:
                    if not hasattr(self, 'var_inputs'): self._load_modbus_config()
                    with self.modbus_lock:
                        # 1. 내부 상태 레지스터 읽기 (기존 270 ~ 289)
                        if hasattr(self.modbus_client, 'get_all_registers'):
                            regs = self.modbus_client.get_all_registers(270, 20)
                            # ★ [방어] 리스트 형태이고 길이가 정확히 20일 때만 통과 (통신에러 튕김 방지)
                            if isinstance(regs, list) and len(regs) == 20:
                                for addr_str, var_name in self.var_inputs.items():
                                    if addr_str.isdigit():
                                        idx = int(addr_str) - 270
                                        if 0 <= idx < len(regs):
                                            val = regs[idx]
                                            self.cached_robot_vars[var_name] = val
                                            self.update_robot_var_ui(var_name, val)
                                            
                        if hasattr(self.modbus_client, 'get_input_register'):
                            in_mask = self.modbus_client.get_input_register(0)   
                            out_mask = self.modbus_client.get_input_register(2)  
                            
                            # ★ [방어] in_mask가 -1이나 65535 같은 쓰레기값이 아닐 때만 처리
                            if isinstance(in_mask, int) and 0 <= in_mask <= 65535:
                                for addr_str, var_name in self.io_inputs.items():
                                    if addr_str.isdigit():
                                        bit_idx = int(addr_str)
                                        if 0 <= bit_idx <= 15:
                                            val = 1 if (in_mask & (1 << bit_idx)) else 0
                                            self.cached_robot_vars[var_name] = val
                                            self.update_robot_var_ui(var_name, val)
                                            
                            # ★ [방어] out_mask 쓰레기값 차단
                            if isinstance(out_mask, int) and 0 <= out_mask <= 65535:
                                for addr_str, var_name in self.io_outputs.items():
                                    if addr_str.isdigit():
                                        bit_idx = int(addr_str) - 32 
                                        if 0 <= bit_idx <= 15:
                                            val = 1 if (out_mask & (1 << bit_idx)) else 0
                                            self.cached_robot_vars[var_name] = val
                                            self.update_robot_var_ui(var_name, val)
                except Exception: pass
                
            try:
                data = self.robot_30001.get_data()
                if data is None:
                    if time.time() - getattr(self, 'last_data_time', 0) > 5.0:
                        if self.check_connection_alive(): self.last_data_time = time.time()
                        else: self.on_robot_disconnected()
                else:
                    self.last_data_time = time.time()
                    self.is_robot_power_on = data.is_robot_power_on
                    self.is_task_running = data.is_task_running
                    self.is_task_paused = data.is_task_paused
                    self.safety_mode = data.safety_mode
                    self.robot_mode = data.robot_mode
                    self.robot_speed = data.target_speed_fraction * 100
                    self.values = (self.is_robot_power_on, self.is_task_running, self.is_task_paused, self.safety_mode, self.robot_mode, self.robot_speed)
                    self.robot_status_update()
                    
                    self.actual_joint_base = data.actual_joint[0]
                    self.actual_joint_shoulder = data.actual_joint[1]
                    self.actual_joint_elbow = data.actual_joint[2]
                    self.actual_joint_wrist1 = data.actual_joint[3]
                    self.actual_joint_wrist2 = data.actual_joint[4]
                    self.actual_joint_wrist3 = data.actual_joint[5]
            except Exception: pass

            now = time.time()
            ing_in = self.cached_robot_vars.get("input_job_ing", 0) == 1
            ing_out = self.cached_robot_vars.get("output_job_ing", 0) == 1
            
            # =======================================================
            # ★ [복구] IO 15번에 맵핑된 진짜 "자동/수동" 작업 모드 감지
            # 원격/로컬 상태와 완전히 분리되어, 오직 UI 갱신용으로만 쓰입니다.
            # =======================================================
            auto_enabled = self.cached_robot_vars.get("자동/수동", 0) == 1
            if auto_enabled != getattr(self, 'auto_mode', False):
                self.auto_mode = auto_enabled
                # 모드가 바뀌었으므로 화면(버튼, 라벨) 즉시 새로고침
                self.robot_status_update()

            # =======================================================
            # ★ [추가] Req 1: 알람 상태에서 Task가 멈춘 것이 확인되면 system_ng OFF
            # =======================================================
            if self.cached_robot_vars.get("system_ng", 0) == 1:
                if not has_error: 
                    self.set_variable_with_ui("system_ng", 0)
                # if not getattr(self, 'is_task_running', True):
                #     self.set_variable_with_ui("system_ng", 0)

            # =======================================================
            # ★ 시작 시퀀스: 초기화 펄스 -> 홈 복귀 -> 완료 확인
            # =======================================================
            current_init_state = getattr(self, 'init_state', "IDLE")
            
            if current_init_state == "REQ_INIT":
                self.set_variable_with_ui("initialize", 1)
                if self.cached_robot_vars.get("initializing", 0) == 1:
                    self.set_variable_with_ui("initialize", 0)
                    self.init_state = "WAIT_INIT_FALSE"
                    
            elif current_init_state == "WAIT_INIT_FALSE":
                if self.cached_robot_vars.get("initializing", 0) == 0:
                    # self.set_variable_with_ui("initialize", 0)
                    print("[SEQ] 로봇 초기화 완료 -> 홈 복귀 시작")
                    self.set_variable_with_ui("home_req", 1) # 홈 명령 투하
                    self.init_state = "WAIT_STARTUP_HOME"
                    
            elif current_init_state == "WAIT_STARTUP_HOME":
                # ★ 1. 출발 감지: 로봇이 홈으로 움직이기 시작하면 즉시 요청 신호 OFF
                if self.cached_robot_vars.get("homing", 0) == 1 and self.cached_robot_vars.get("home_req", 0) == 1:
                    self.set_variable_with_ui("home_req", 0)
                    print("[SEQ] 시작 시퀀스: 로봇 Homing 시작 감지 -> home_req 신호 OFF")
                    
                # ★ 2. 도착 감지: 홈 위치에 완전히 도달했을 때
                is_home = self.cached_robot_vars.get("is_home", 0) == 1
                if is_home:
                    if self.cached_robot_vars.get("home_req", 0) == 1:
                        self.set_variable_with_ui("home_req", 0)
                        
                    print("[SEQ] 홈 복귀 완료 -> GODO 지시 수락 준비 완료")
                    self.is_startup_ready = True # 이제부터 진짜 작업 가능
                    self.init_state = "IDLE"
                    
                    # =======================================================
                    # ★ [누락 로직 복구] 교정 모드일 경우 홈 도착 즉시 교정 시퀀스 돌입!
                    # =======================================================
                    if self.ui.calibration_mode_btn.isChecked() and hasattr(self, 'calib_target_cell'):
                        area = self.calib_target_cell
                        if area in [1, 2]:
                            with self.state_lock:
                                self.cell_data[area]["seq_state"] = "WAIT_CALIB_START"
                                self.cell_data[area]["req_sent"] = False
                            print(f"[SEQ] 교정 선원 모드: 셀 {area} 교정 자동 시퀀스 진입 완료")
                    
            if self.cached_robot_vars.get("home_req", 0) == 1 and self.cached_robot_vars.get("homing", 0) == 1:
                self.set_variable_with_ui("home_req", 0)
                print("[SEQ] 로봇 Homing 시작 감지 -> home_req 신호 즉시 OFF")

            # =======================================================
            # 1. IO 기반 상태 감지 (100ms 노이즈 필터링 적용)
            # =======================================================
            is_emg = self.safety_mode in [6, 7]
            
            # 물리적 RAW 신호 읽기
            raw_cyl1_error = self.cached_robot_vars.get("실린더1 알람", 1) == 0
            raw_cyl2_error = self.cached_robot_vars.get("실린더2 알람", 1) == 0
            
            curr_time = time.time()
            
            # --- 실린더 1 필터링 ---
            if raw_cyl1_error:
                if getattr(self, 'cyl1_error_start', 0) == 0:
                    self.cyl1_error_start = curr_time # 에러 시작 시간 기록
                cyl1_alarm = (curr_time - self.cyl1_error_start >= 0.5) # 500ms 유지 확인
            else:
                self.cyl1_error_start = 0
                cyl1_alarm = False
                
            # --- 실린더 2 필터링 ---
            if raw_cyl2_error:
                if getattr(self, 'cyl2_error_start', 0) == 0:
                    self.cyl2_error_start = curr_time
                cyl2_alarm = (curr_time - self.cyl2_error_start >= 0.1)
            else:
                self.cyl2_error_start = 0
                cyl2_alarm = False

            # --- 알람 발생 및 리셋 트리거 ---
            if cyl1_alarm and not getattr(self, 'cyl1_alarm_triggered', False):
                self.report_alarm_signal.emit("실린더 1 알람 발생 (리셋 필요)", "ERROR")
                self.cyl1_alarm_triggered = True
            elif not cyl1_alarm: 
                self.cyl1_alarm_triggered = False

            if cyl2_alarm and not getattr(self, 'cyl2_alarm_triggered', False):
                self.report_alarm_signal.emit("실린더 2 알람 발생 (리셋 필요)", "ERROR")
                self.cyl2_alarm_triggered = True
            elif not cyl2_alarm: 
                self.cyl2_alarm_triggered = False
            
            if not self.robot_disconnected:
                auto_enabled = self.cached_robot_vars.get("자동/수동", 0) == 1
                if auto_enabled != getattr(self, 'auto_mode', False):
                    self.auto_mode = auto_enabled
            
            has_error = is_emg or cyl1_alarm or cyl2_alarm
            
            # ★ 새로운 알람 발생 시 부저 뮤트 자동 해제 (자동 initialize 송출 로직 제거됨)
            if has_error and getattr(self, 'last_lamp_state', None) != "ALARM":
                self.is_buzzer_muted = False
                
            current_lamp_state = "ALARM" if has_error else ("AUTO" if self.auto_mode else "MANUAL")
            
            if (current_lamp_state != getattr(self, 'last_lamp_state', None)) or \
               (has_error and getattr(self, '_last_buzzer_mute_state', False) != getattr(self, 'is_buzzer_muted', False)):
                
                self.last_lamp_state = current_lamp_state
                self._last_buzzer_mute_state = getattr(self, 'is_buzzer_muted', False)
                
                # if has_error: 
                #     buzz_val = 0 if self.is_buzzer_muted else 1
                #     self.set_lamps(red=1, yellow=0, green=0, buzz=buzz_val)
                # elif self.auto_mode: 
                #     self.set_lamps(red=0, yellow=0, green=1, buzz=0)
                # else: 
                #     self.set_lamps(red=0, yellow=1, green=0, buzz=0)

            # =======================================================
            # ★ 2. 시퀀스 엔진 구동 (단일 루프로 통합!)
            # =======================================================
            with self.state_lock:
                for area in [1, 2]:
                    cell = self.cell_data[area]
                    state = cell["seq_state"]
                    # print(f"[DEBUG] currnet sequence state: {state}")
                    
                    # 로봇 현재 상태 한 번에 싹 읽어오기
                    in_done = self.cached_robot_vars.get("input_job_done", 0) == 1
                    out_done = self.cached_robot_vars.get("output_job_done", 0) == 1
                    is_home = self.cached_robot_vars.get("is_home", 0) == 1
                    sensor_arrived = self.cached_robot_vars.get("cell_sensor_arrived", 0) == 1
                    ing_in = self.cached_robot_vars.get("input_job_ing", 0) == 1
                    ing_out = self.cached_robot_vars.get("output_job_ing", 0) == 1
                    
                    # -----------------------------------------------
                    # [Input 투입 시퀀스]
                    # -----------------------------------------------
                    if state == "WAIT_INPUT_START":
                        if not cell.get("req_sent", False):
                            curr_pt = cell.get("current_point", 1)
                            self.set_variable_with_ui(f"J{area}_target_point", curr_pt)
                            self.set_variable_with_ui(f"c{area}_input_job_req", 1)
                            cell["req_sent"] = True  
                            print(f"[SEQ] 셀 {area} INPUT 요청 전송 -> 로봇 출발 응답 대기 중...")
                        
                        if ing_in or sensor_arrived:
                            self.set_variable_with_ui(f"c{area}_input_job_req", 0)
                            cell["req_sent"] = False 
                            cell["seq_state"] = "INPUT_ING"
                            print(f"[DEBUG] 셀 {area} INPUT 출발 인지 완료! -> INPUT_ING 상태 진입")
                            
                    elif state == "INPUT_ING":
                        if sensor_arrived:
                            if cell.get("sensor_arrive_time", 0) == 0:
                                cell["sensor_arrive_time"] = time.time()
                                print(f"[DEBUG] 셀 {area} 센서 도착! 카메라 안정화 대기 (0.5s)")
                                
                            elif time.time() - cell["sensor_arrive_time"] >= 0.5:
                                cell["sensor_arrive_time"] = 0
                                cell["seq_state"] = "VISION_MEASURING"
                                print(f"[DEBUG] 셀 {area} 비전 촬영 명령 투하!")
                                self.start_vision_task(area, is_calib=False)
                                
                    elif state == "WAIT_IN_FINISH":
                        # 비전 검사까지 모두 끝나고 센서 위치를 벗어난 후, 최종 완료 신호를 확인하는 곳입니다.
                        if in_done:
                            print(f"[SEQ] 셀 {area} 투입 최종 완료 -> Home 복귀")
                            self.set_variable_with_ui("sensor_done", 0)
                            self.set_variable_with_ui("home_req", 1)
                            cell["seq_state"] = "WAIT_IS_HOME"

                    # -----------------------------------------------
                    # [Output 배출 시퀀스]
                    # -----------------------------------------------
                    elif state == "WAIT_OUTPUT_START":
                        if not cell.get("req_sent", False):
                            curr_pt = cell.get("current_point", 1)
                            self.set_variable_with_ui(f"J{area}_target_point", curr_pt)
                            self.set_variable_with_ui(f"c{area}_output_job_req", 1)
                            cell["req_sent"] = True
                            cell["wait_out_time"] = time.time() # 타임스탬프 기록
                        
                        # ★ [방어 3] 과거 잔류 out_done 신호에 속지 않기 위한 0.5초 방어막 추가
                        if ing_out or (out_done and time.time() - cell.get("wait_out_time", 0) > 0.5):
                            self.set_variable_with_ui(f"c{area}_output_job_req", 0)
                            cell["req_sent"] = False
                            cell["seq_state"] = "OUTPUT_ING"
                            print(f"[DEBUG] 셀 {area} OUTPUT 출발 인지 완료! -> OUTPUT_ING 상태 진입")

                    elif state == "OUTPUT_ING":
                        if out_done:
                            print(f"[SEQ] 셀 {area} 배출 완료 -> Home 복귀")
                            self.set_variable_with_ui("home_req", 1)
                            cell["seq_state"] = "WAIT_IS_HOME"

                    # -----------------------------------------------
                    # [Calibration 교정 선원 시퀀스]
                    # -----------------------------------------------
                    elif state == "WAIT_CALIB_START":
                        if not cell.get("req_sent", False):
                            # 수동 조작부와 동일하게 J=0, input=1, sensor=1 콤보로 쏩니다.
                            self.set_variable_with_ui(f"J{area}_target_point", 0)
                            self.set_variable_with_ui(f"c{area}_input_job_req", 1)
                            self.set_variable_with_ui(f"c{area}_sensor_req", 1)
                            cell["req_sent"] = True
                            print(f"[SEQ] 셀 {area} 교정 위치(센서) 이동 지시 발송 -> 출발 대기...")
                        
                        # 로봇이 출발(ing_in)하거나 센서에 도착하면 즉시 요구 신호 OFF (Pulse 구현)
                        if ing_in or sensor_arrived:
                            self.set_variable_with_ui(f"c{area}_input_job_req", 0)
                            self.set_variable_with_ui(f"c{area}_sensor_req", 0)
                            cell["req_sent"] = False
                            cell["seq_state"] = "CALIB_ING"
                            print(f"[DEBUG] 셀 {area} 교정 출발 인지 완료! -> CALIB_ING 상태 진입")

                    elif state == "CALIB_ING":
                        if sensor_arrived:
                            if cell.get("sensor_arrive_time", 0) == 0:
                                cell["sensor_arrive_time"] = time.time()
                                print(f"[SEQ] 셀 {area} 교정 센서 도착! 안정화 대기 (0.5s)")
                                
                            elif time.time() - cell["sensor_arrive_time"] >= 0.5:
                                cell["sensor_arrive_time"] = 0
                                cell["seq_state"] = "VISION_MEASURING"
                                print(f"[SEQ] 셀 {area} 교정 촬영 명령 투하!")
                                self.start_vision_task(area, is_calib=True)

                    elif state == "WAIT_CALIB_EXIT":
                        # 로봇이 sensor_done을 받고 센서 위치에서 빠져나갔을 때(False가 되었을 때)
                        if not sensor_arrived:
                            print(f"[SEQ] 셀 {area} 센서 위치 이탈 감지 -> 홈 복귀 요청")
                            self.set_variable_with_ui("sensor_done", 0)
                            self.set_variable_with_ui("home_req", 1)
                            cell["seq_state"] = "WAIT_IS_HOME"

                    # -----------------------------------------------
                    # [공통 홈 복귀 대기]
                    # -----------------------------------------------
                    elif state == "WAIT_IS_HOME" and is_home:
                        # ★ 이미 homing 시작 시점에 꺼졌으므로 불필요한 통신 삭제
                        print(f"[SEQ] 셀 {area} 홈 복귀 완료 ➔ IDLE")
                        cell["seq_state"] = "IDLE"
                        
                        if cell["last_action"] == "output":
                            curr_pt = cell.get("current_point", 1)
                            if hasattr(self, 'db_manager'): 
                                self.db_manager.finish_job(area, curr_pt)
                            self.increment_workload(area)
                            
                        # =======================================================
                        # ★ 교정 모드 완료 후 '작업 정지' 자동 클릭 처리
                        # =======================================================
                        if self.ui.calibration_mode_btn.isChecked():
                            print(f"[SEQ] 셀 {area} 교정 작업 완전 종료 -> 프로그램 자동 정지(Stop) 실행")
                            QtCore.QTimer.singleShot(100, self.on_stop_button_clicked)

        except Exception as e:
            import traceback
            traceback.print_exc()

        finally:
            self._is_polling = False
            
    def get_current_joints(self):
        """현재 로봇의 6축 관절 각도(Base, Shoulder, Elbow, Wrist1, 2, 3)를 리스트로 반환"""
        return [
            getattr(self, 'actual_joint_base', 0.0),
            getattr(self, 'actual_joint_shoulder', 0.0),
            getattr(self, 'actual_joint_elbow', 0.0),
            getattr(self, 'actual_joint_wrist1', 0.0),
            getattr(self, 'actual_joint_wrist2', 0.0),
            getattr(self, 'actual_joint_wrist3', 0.0)
        ]
            
    # -------------------------------------------------------------------
    # ★ 하드웨어 IO 제어 (램프, 부저, 알람 리셋)
    # -------------------------------------------------------------------
    def set_lamps(self, red, yellow, green, buzz):
        self.set_variable_with_ui("Lamp-R", red)
        self.set_variable_with_ui("Lamp-Y", yellow)
        self.set_variable_with_ui("Lamp-G", green)
        self.set_variable_with_ui("Buzzer", buzz)

    def save_tcp_csv(self, msg):
        now_str = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz")
        time_only = now_str.split(" ")[1]

        if "📥 [수신]" in msg:
            direction = "수신"
            data = msg.replace("📥 [수신]", "").strip()
            msg_color = "#4CAF50" 
        elif "📤 [송신]" in msg:
            direction = "송신"
            data = msg.replace("📤 [송신]", "").strip()
            msg_color = "#2196F3" 
        else:
            return 
            
        if hasattr(self.ui, 'godo_msg_area'):
            html_msg = f"<span style='color: {msg_color};'>[{time_only}] {msg.strip()}</span>"
            
            # ★ 핵심: 직접 UI를 건드리지 않고, 방금 만든 대리자(append_godo_msg_safe)에게 위임합니다!
            QtCore.QMetaObject.invokeMethod(self, "append_godo_msg_safe", Qt.QueuedConnection, QtCore.Q_ARG(str, html_msg))
            
        # (파일 저장 로직은 기존과 동일)
        if hasattr(self, 'godo_log_path'):
            plain_msg = f"[{time_only}] [{direction}] {data}"
            self.prepend_log_to_txt(self.godo_log_path, plain_msg)
            
    # ★ [신규 추가] Print 출력들을 system_msg_area에 쏴주는 함수
    @pyqtSlot(str)
    def append_system_msg(self, text):
        if hasattr(self.ui, 'system_msg_area'):
            time_str = QDateTime.currentDateTime().toString("HH:mm:ss.zzz")
            html_msg = f"<span>[{time_str}] {text.strip()}</span>"
            
            self.safe_append_log(self.ui.system_msg_area, html_msg)
            
            # ★ [신규] 최신 기록이 위로 가도록 txt 파일에 실시간 저장
            plain_msg = f"[{time_str}] {text.strip()}"
            self.prepend_log_to_txt(self.sys_log_path, plain_msg)

    @pyqtSlot(str, str)
    def append_vision_logs(self, qr_msg, vis_msg):
        if hasattr(self.ui, 'qr_result_area'):
            self.safe_append_log(self.ui.qr_result_area, qr_msg)
        if hasattr(self.ui, 'final_result_area'):
            self.safe_append_log(self.ui.final_result_area, vis_msg)
            
    @pyqtSlot()
    def update_ui_state(self):
        """메인 스레드에서 안전하게 UI의 모든 숫자와 라벨을 새로고침합니다."""
        try:
            for cell_num in [1, 2]:
                cell = self.cell_data[cell_num]
                
                # 목표 수량/번호 결정
                t_points = cell.get("target_points", [])
                if t_points:
                    goal_str = ", ".join(map(str, t_points))
                else:
                    goal_str = str(cell.get("target", 0))

                # 완료 수량/번호 결정
                d_points = cell.get("done_points", [])
                if d_points:
                    done_str = ", ".join(map(str, d_points))
                else:
                    # ★ 배열이 비어있으면 강제로 현재 카운트(pos)를 찍어줌 (둘 다 0이면 "0" 표시)
                    done_str = str(cell.get("pos", 0))

                # UI 라벨에 적용 (객체 존재 여부 확인 필수)
                goal_label = getattr(self.ui, f"cell_{cell_num}_work_goal_label", None)
                if goal_label:
                    goal_label.setText(goal_str)
                    
                done_label = getattr(self.ui, f"cell_{cell_num}_work_done_label", None)
                if done_label:
                    done_label.setText(done_str)
                
                # 작업 모드 텍스트 갱신
                work_label = getattr(self.ui, f"cell_{cell_num}_work", None)
                if work_label:
                    mode_str = cell.get("work_mode", "순차 작업")
                    work_label.setText(f"셀 {cell_num} 작업 ({mode_str})")

            # 리셋 후 버튼 상태 즉시 동기화
            self.robot_status_update()

        except Exception as e:
            print(f"[UI UPDATE ERROR] {e}")
    
    def start_vision_task(self, area, is_calib=False):
        """비전 측정을 상주하는 브릿지 큐로 던집니다."""
        beaker_name = self.current_beaker_name_cached
        jig_num = self.cell_data[area].get("current_point", 0) if area in self.cell_data else 0
        if getattr(self, 'is_vision_ready', False):
            self.vision_bridge.request_task(area, is_calib, beaker_name, jig_num)
        else:
            self.on_vision_result_ready(area, -1.0, -1.0, -1.0, 0.0, "FAIL", is_calib)

    @pyqtSlot(int, float, float, float, float, str, bool)
    def on_vision_result_ready(self, area, cx, cy, cr, b_depth, qr_data, is_calib):
        
        # 수동 측정(테스트)인 경우
        if area == 999:
            if cx != -1:
                msg = (f"<b>[비전 측정 성공]</b><br><br>"
                       f"<b>QR 데이터:</b> {qr_data}<br>"
                       f"<b>중심 좌표:</b> X: {cx} / Y: {cy}<br>"
                       f"<b>반지름:</b> {cr} px<br>"
                       f"<b>깊이 (Z):</b> {b_depth * 1000:.1f} mm")
                self.show_message("테스트 결과", msg, QMessageBox.Information, QMessageBox.Ok)
            else:
                self.show_message("테스트 실패", "비커 중심 또는 QR 코드를 찾지 못했습니다.", QMessageBox.Warning, QMessageBox.Ok)
            return

        cell = self.cell_data[area]
        curr_pt = cell.get("current_point", 1)
        
        # =====================================================================
        # ★ 메인 스레드 렉(프리징) 완벽 차단:
        # DB 저장, JSON 파일 읽기, Modbus 통신 등 무거운 작업을 백그라운드 스레드로 분리
        # =====================================================================
        def _process_result_bg():
            nonlocal qr_data # 내부에서 값을 변경할 수 있도록 선언
            
            c_loc = b_loc = loc_dev = c_depth = b_depth_str = d_dev = "N/A"
            ref_data = None
            
            # =================================================================
            # ★ [신규] product.json에서 현재 선택된 비커의 교정값을 실시간으로 로드
            # =================================================================
            try:
                config_path = PRODUCT_PATH / "product.json"
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        p_data = json.load(f)
                    
                    calib_area_data = p_data.get("calibration", {}).get(self.current_beaker_name_cached, {}).get(f"area{area}")
                    if calib_area_data:
                        ref_data = (calib_area_data["cx"], calib_area_data["cy"], calib_area_data["depth"])
            except Exception: 
                pass
            
            if not is_calib:
                if cx != -1:
                    b_loc = f"({cx:.1f},{cy:.1f})"
                    b_depth_str = f"{b_depth:.3f}m"
                
                # 교정값(ref_data)이 존재할 때만 편차 계산
                if ref_data and cx != -1:
                    ref_cx, ref_cy, ref_depth = ref_data
                    c_loc = f"({ref_cx:.1f},{ref_cy:.1f})"
                    
                    pixel_dist = math.sqrt((cx - ref_cx)**2 + (cy - ref_cy)**2)
                    
                    physical_diameter_mm = 0
                    try:
                        if config_path.exists():
                            with open(config_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                beaker_data = data.get("beaker", {}).get(self.current_beaker_name_cached, {})
                                physical_diameter_mm = beaker_data.get("diameter", beaker_data.get("target_mm", 0))
                    except: pass
                    
                    if physical_diameter_mm > 0 and cr > 0:
                        mm_per_px = (physical_diameter_mm / 2.0) / float(cr)
                        real_dist_mm = pixel_dist * mm_per_px
                        loc_dev = f"{real_dist_mm:.1f}mm"
                    else:
                        loc_dev = f"{pixel_dist:.1f}px"
                        
                    c_depth = f"{ref_depth:.3f}m"
                    d_dev = f"{abs(b_depth - ref_depth):.3f}m"
                
                if qr_data == "FAIL" or qr_data == "NOT_FOUND":
                    qr_data = "ReadFail"
                    cell["serial"] = cell.get('serial', 1) + 1
                
                # ★ DB 저장 (무거운 디스크 I/O 처리)
                if hasattr(self, 'db_manager'):
                    current_joints = str([round(j, 4) for j in self.get_current_joints()])
                    self.db_manager.update_job_vision_data(area, curr_pt, qr_data, current_joints, c_loc, b_loc, loc_dev, c_depth, b_depth_str, d_dev)

                # ★ 상태 업데이트
                with self.state_lock:
                    self.cell_data[area]["sample_id"] = qr_data
                    self.cell_data[area]["calib_loc"] = c_loc
                    self.cell_data[area]["beaker_loc"] = b_loc
                    self.cell_data[area]["loc_dev"] = loc_dev
                    self.cell_data[area]["calib_depth"] = c_depth
                    self.cell_data[area]["beaker_depth"] = b_depth_str
                    self.cell_data[area]["depth_dev"] = d_dev
                    self.cell_data[area]["seq_state"] = "WAIT_IN_FINISH"

                print(f"[DEBUG] 셀 {area} 비전 검사 완료 -> sensor_done ON 송신")
                self.set_variable_with_ui("sensor_done", 1)

                time_str = QDateTime.currentDateTime().toString("HH:mm:ss")
                qr_msg = f"<span style='color: #E040FB;'>[{time_str}] 셀 {area} QR : {qr_data}</span>"
                if cx != -1:
                    vis_msg = (f"<span style='color: #4CAF50;'>[{time_str}] 셀 {area} 측정 완료<br>"
                            f"&nbsp;&nbsp;▶ 좌표 : X={cx:.1f}, Y={cy:.1f}<br>"
                            f"&nbsp;&nbsp;▶ 깊이 : {b_depth*1000:.1f} mm<br>")
                else:
                    vis_msg = f"<span style='color: #F44336;'>[{time_str}] 셀 {area} : 비전 검출 실패 (FAIL)</span><br>"

                QtCore.QMetaObject.invokeMethod(self, "append_vision_logs", Qt.QueuedConnection, QtCore.Q_ARG(str, qr_msg), QtCore.Q_ARG(str, vis_msg))

            else:
                # =================================================================
                # ★ [Calibration 교정 비전 처리] product.json 저장 명령 호출
                # =================================================================
                self.set_variable_with_ui(f"c{area}_sensor_req", 0)
                if cx != -1 or qr_data != "FAIL":
                    loc_str = f"({cx:.1f},{cy:.1f})"
                    depth_str = f"{b_depth:.3f}m"
                    
                    if hasattr(self, 'vision_bridge') and hasattr(self.vision_bridge, 'hw_worker'):
                        # ★ 현재 비커 이름을 넘겨주어 귀속되도록 저장합니다.
                        self.vision_bridge.hw_worker.core.save_calibration_data(area, cx, cy, b_depth, self.current_beaker_name_cached)
                    
                    if hasattr(self, 'db_manager') and hasattr(self.db_manager, 'update_calibration_data_all_points'):
                        self.db_manager.update_calibration_data_all_points(area, loc_str, depth_str)
                        
                    print(f"[SEQ] 셀 {area} 기준점 교정 완벽 갱신/저장 완료: {loc_str}, Depth={depth_str}, QR={qr_data}")
                else:
                    print(f"[SEQ WARN] 셀 {area} 기준점 교정 실패: 비전을 찾지 못했습니다.")
                
                self.set_variable_with_ui("sensor_done", 1)
                
                with self.state_lock: 
                    self.cell_data[area]["seq_state"] = "WAIT_CALIB_EXIT"

            QtCore.QMetaObject.invokeMethod(self, "update_ui_state", Qt.QueuedConnection)

        threading.Thread(target=_process_result_bg, daemon=True).start()

    def increment_workload(self, cell_num):
        cell = self.cell_data[cell_num]
        
        if cell.get("target_points") and len(cell.get("done_points", [])) < len(cell["target_points"]):
            done_pt = cell["target_points"][len(cell["done_points"])]
            cell["done_points"].append(done_pt)
            
        cell["pos"] += 1
        self.db_manager.update_work_done(str(cell_num), cell["pos"])
        
        # 완료 배열 DB 저장
        if hasattr(self.db_manager, 'update_work_done_points'):
            self.db_manager.update_work_done_points(str(cell_num), cell["done_points"])
            
        self.update_ui_state()
        self.check_and_auto_stop()
        
    def check_and_auto_stop(self):
        """양쪽 셀 모두 투입할 시료(남은 목표 수량)가 없는지 확인하고, 없다면 자동 정지합니다."""
        if not getattr(self, 'is_working', False):
            return False
            
        all_empty = True
        for area in [1, 2]:
            cell = self.cell_data[area]
            target_count = cell.get("target", 0)
            done_count = len(cell.get("done_points", []))
            
            if target_count > 0 and done_count < target_count:
                all_empty = False
                break
                
        if all_empty:
            print("[SYSTEM] 양쪽 셀 모두 남은 작업(투입할 시료)이 없습니다. 작업을 자동 종료합니다.")
            self.is_working = False 
            QtCore.QMetaObject.invokeMethod(self.ui.stop_btn, "click", Qt.QueuedConnection)
            return True
            
        return False

    def reset_workload(self):
        # ★ 파라미터 이름(icon, buttons)을 명시적으로 지정하여 팝업 띄우기
        reply = self.show_message(
            title="작업량 초기화", 
            message="두 셀의 작업 목표 및 완료 수량을 모두 <b>0으로 초기화</b>하시겠습니까?", 
            icon=QMessageBox.Question,
            buttons=QMessageBox.Yes | QMessageBox.No, 
            font_size=20
        )
        
        if reply == QMessageBox.Yes:
            self.update_log_paths(is_reset=True) # ★ 추가됨: 리셋 시 파일명 번호 증가
            for cell_num in [1, 2]:
                self.cell_data[cell_num]["target"] = 0
                self.cell_data[cell_num]["pos"] = 0
                self.cell_data[cell_num]["target_points"] = []
                self.cell_data[cell_num]["done_points"] = []
                self.cell_data[cell_num]["last_action"] = "none"
                self.cell_data[cell_num]["seq_state"] = "IDLE"
                
                current_beaker = self.ui.current_beaker.text()
                try:
                    self.db_manager.setup_new_work_by_cell(
                        cell_num, 0, beaker_type=current_beaker,
                        work_mode=self.cell_data[cell_num]["work_mode"],
                        target_points=[]
                    )
                    self.db_manager.update_work_done(str(cell_num), 0)
                except Exception as e:
                    print(f"[DB WARN] DB 저장 예외 발생: {e}")
                
            self.update_ui_state()
            self.check_and_auto_stop()
            print("[SYSTEM] 작업 목표 및 완료 수량이 완벽하게 0으로 초기화되었습니다.")
            
    @pyqtSlot()
    def headless_reset_all(self):
        """TCP 스레드의 간섭 없이 메인 스레드에서 양쪽 셀(1, 2)을 동시에 완벽 리셋합니다."""
        print("[RESET START] 전체(셀 1, 2) 통합 초기화 프로세스 시작...")
        try:
            self.update_log_paths(is_reset=True)
            
            # 1. 두 셀의 내부 변수 모두 0으로 초기화
            with self.state_lock:
                for cell_num in [1, 2]:
                    self.cell_data[cell_num]["target"] = 0
                    self.cell_data[cell_num]["pos"] = 0
                    self.cell_data[cell_num]["target_points"] = []
                    self.cell_data[cell_num]["done_points"] = []
                    self.cell_data[cell_num]["last_action"] = "none"
                    self.cell_data[cell_num]["seq_state"] = "IDLE"
                    self.cell_data[cell_num]["current_point"] = 0
                    self.cell_data[cell_num]["sample_id"] = ""
                
            current_beaker = self.ui.current_beaker.text()
            
            # 2. 두 셀의 DB 테이블 모두 0으로 덮어쓰기
            if hasattr(self, 'db_manager'):
                for cell_num in [1, 2]:
                    self.db_manager.setup_new_work_by_cell(
                        cell_num, 0, beaker_type=current_beaker,
                        work_mode=self.cell_data[cell_num].get("work_mode", "순차 작업"),
                        target_points=[]
                    )
                    self.db_manager.update_work_done(str(cell_num), 0)
                    if hasattr(self.db_manager, 'update_work_done_points'):
                        self.db_manager.update_work_done_points(str(cell_num), [])
            
            # 3. 화면 UI 강제 갱신 (0, 0으로 즉시 바뀜)
            self.update_ui_state()
            
            # 4. 시스템 로그 기록
            reset_msg = "<span style='color: #FF9800;'>[SYSTEM] 외부 고도(GODO) 요청으로 전체 작업(셀 1, 2) 및 DB 초기화 완료</span>"
            if hasattr(self.ui, 'system_msg_area'):
                self.safe_append_log(self.ui.system_msg_area, reset_msg)
            if hasattr(self.ui, 'godo_msg_area'):
                self.safe_append_log(self.ui.godo_msg_area, reset_msg)
            
            print("[RESET SUCCESS] 전체 작업 목표 및 완료 수량이 완벽하게 0으로 초기화되었습니다.")
            
        except Exception as e:
            print(f"[RESET ERROR] 전체 리셋 중 치명적 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
    def reset_done_workload(self):
        """현재까지 완료된 작업 번호와 수량만 0으로 리셋합니다."""
        reply = self.show_message(
            title="완료 작업 리셋", 
            message="현재까지 진행된 <b>완료 수량과 포인트 기록</b>만 초기화하시겠습니까?<br>(설정된 목표 수량은 유지됩니다)", 
            icon=QMessageBox.Question,
            buttons=QMessageBox.Yes | QMessageBox.No, 
            font_size=20
        )
        
        if reply == QMessageBox.Yes:
            with self.state_lock:
                for area in [1, 2]:
                    self.cell_data[area]["pos"] = 0
                    self.cell_data[area]["done_points"] = []
                    self.cell_data[area]["last_action"] = "none"
                    self.cell_data[area]["seq_state"] = "IDLE"
                    
                    if hasattr(self, 'db_manager'):
                        try:
                            self.db_manager.update_work_done(str(area), 0)
                            self.db_manager.update_work_done_points(str(area), [])
                            # ★ 완료 리셋 시 DB 슬롯(result)도 깔끔하게 '-' 와 'waiting'으로 재배치
                            self.db_manager.setup_new_work_by_cell(
                                area, self.cell_data[area]["target"],
                                beaker_type=self.ui.current_beaker.text(),
                                work_mode=self.cell_data[area]["work_mode"],
                                target_points=self.cell_data[area]["target_points"]
                            )
                        except Exception as e:
                            print(f"[DB WARN] 완료 수량 리셋 중 오류: {e}")
            
            self.update_ui_state()
            self.check_and_auto_stop()
            print("[SYSTEM] 완료된 작업 기록이 초기화되었습니다.")
            
    def show_job_result_popup(self, cell_num, point_num):
        if self.active_popups.get(cell_num) is not None:
            try: self.active_popups[cell_num].close()
            except: pass
        print(f"[UI] 작업 완료 팝업 생성 (Cell {cell_num}, Point {point_num})")
        dlg = JobResultDialog(self, cell_num, point_num)
        dlg.show()
        self.active_popups[cell_num] = dlg

    # -------------------------------------------------------------------
    # 기존 오리지널 UI/연결/수동 조작 이벤트 처리 로직 (원상 복구됨)
    # -------------------------------------------------------------------
    def load_robot_config(self):
        if not os.path.exists(self.robot_config_file):
            default_config = {"ROBOT_IP": "192.168.227.130", "ROBOT_PORT_1": 29999, "ROBOT_PORT_2": 30001, "ROBOT_PORT_3": 502}
            with open(self.robot_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        with open(self.robot_config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def update_robot_var_ui(self, var_name, value):
        self.update_var_signal.emit(var_name, value)
        
    @pyqtSlot(str, object)
    def _do_update_robot_var_ui(self, var_name, value):
        if not hasattr(self, 'var_item_map'): return
        val_item = self.var_item_map.get(var_name)
        if val_item:
            if isinstance(value, bool): display_text = "1" if value else "0"
            elif str(value).strip().lower() == "true": display_text = "1"
            elif str(value).strip().lower() == "false": display_text = "0"
            elif isinstance(value, float):
                if value.is_integer(): display_text = str(int(value))
                else: display_text = f"{value:.4f}"
            else: display_text = str(value)
            val_item.setText(display_text)
            
    def create_robot_objects(self):
        if hasattr(self, 'robot_30001') and self.robot_30001:
            try: self.robot_30001.__sock.close() 
            except: pass
        if hasattr(self, 'robot_29999') and self.robot_29999:
            try: self.robot_29999.sock.close()
            except: pass
        if hasattr(self, 'modbus_client') and self.modbus_client is not None:
            try: self.modbus_client.disconnect()
            except: pass
        self.robot_30001 = Robot_30001(self.robot_ip, self.robot_port2)
        self.robot_29999 = Robot_29999(self.robot_ip, self.robot_port1)
        self.modbus_client = Robot_modbus(host=self.robot_ip, port=self.modbus_port)

    def start_background_connection(self):
        print("[System] 백그라운드 자동 연결 시작...")
        self.connect_robot(show_popup=False)

    def connect_robot(self, show_popup=False):
        if hasattr(self, 'conn_thread') and self.conn_thread.isRunning(): return
        self.show_popup_on_fail = show_popup
        self.robot_disconnected = True 
        self.create_robot_objects()
        self.conn_thread = ConnectionThread(self.robot_29999, self.robot_30001, self.modbus_client)
        self.conn_thread.finished_signal.connect(self.on_connect_finished)
        self.conn_thread.start()
        
    def on_connect_finished(self, success, message):
        if success:
            self.robot_disconnected = False
            
            # ★ 핵심 1: 타이머 초기화 (데이터 수신을 위한 5초 유예기간 부여)
            # 이 코드가 없으면 연결되자마자 과거의 시간 차이 때문에 스스로 다시 끊어버립니다!
            self.last_data_time = time.time() 
            
            print(f"[INFO] 로봇 연결 성공 ({self.robot_ip})")
            self.init_robot_var_ui()
            self.send_product_offsets_to_robot()
            
            # ★ 핵심 2: UI 즉시 새로고침
            # 첫 통신 데이터가 도착하기 전 찰나의 순간에도 '정상'으로 표시되게끔 임시 값 주입
            self.robot_mode = 7     # 7: 전원 켜짐
            self.safety_mode = 1    # 1: 정상 상태
            self.robot_status_update() # 화면 강제 갱신 명령!
            
            if getattr(self, 'alarm_thread', None) is None or not self.alarm_thread.isRunning():
                self.alarm_thread = AlarmThread(self, self.alarm_manager)
                self.alarm_thread.alarm_received_signal.connect(self.add_alarm_log)
                self.alarm_thread.start()
        else:
            self.robot_disconnected = True
            
            # ★ 연결 실패 시에도 화면을 갱신해서 회색으로 확실히 굳혀줌
            self.robot_status_update() 
            
            print(f"[ERROR] 로봇 연결 실패: {message}")
            if getattr(self, 'show_popup_on_fail', False):
                self.show_message("연결 실패", f"로봇 연결에 실패했습니다.<br>IP: {self.robot_ip}<br>Error: {message}", QMessageBox.Critical)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            if obj == self.ui.cell_1_work_goal_label:
                self.show_work_setup_dialog(cell_num=1)
                return True
            elif obj == self.ui.cell_2_work_goal_label:
                self.show_work_setup_dialog(cell_num=2)
                return True
            # ★ 추가: 카메라 화면을 더블클릭하면 수동 비전 테스트 실행
            elif hasattr(self.ui, 'label_cam') and obj == self.ui.label_cam:
                self.run_manual_vision_test()
                return True
        return super().eventFilter(obj, event)
    
    @pyqtSlot(QtGui.QImage)
    def update_camera_image(self, qimg):
        """이미지를 UI 라벨 크기에 맞춰 고속으로 표시합니다."""
        if hasattr(self.ui, 'label_cam') and self.ui.label_cam:
            label_w = self.ui.label_cam.width()
            label_h = self.ui.label_cam.height()
            
            pixmap = QtGui.QPixmap.fromImage(qimg)
            
            # ★ SmoothTransformation(고화질/고부하) -> FastTransformation(고속/저부하) 변경
            scaled_pixmap = pixmap.scaled(
                label_w, 
                label_h, 
                Qt.KeepAspectRatio, 
                Qt.FastTransformation 
            )
            
            self.ui.label_cam.setAlignment(Qt.AlignCenter)
            self.ui.label_cam.setPixmap(scaled_pixmap)
    
    def run_manual_vision_test(self):
        """카메라 화면 더블클릭 시 실행되는 비전 수동 테스트 함수"""
        if not getattr(self, 'is_vision_ready', False):
            self.show_message("테스트 불가", "비전 매니저가 초기화되지 않았거나 카메라 연결에 실패했습니다.", QMessageBox.Warning)
            return

        print("[TEST] 수동 비전 측정 테스트 시작...")
        reply = self.show_message(
            title="비전 수동 테스트", 
            message="카메라 측정을 시작하시겠습니까?\n약 1~2초 정도 소요됩니다.", 
            icon=QMessageBox.Question,
            buttons=QMessageBox.Yes | QMessageBox.No, 
            font_size=20
        )
        
        if reply == QMessageBox.Yes:
            QApplication.processEvents()
            # 워커 스레드로 수동 테스트 지시
            self.vision_bridge.request_task(999, is_calib=False)
    
    def show_work_setup_dialog(self, cell_num):
        dialog = WorkSetupDialog(self, cell_num)
        dialog.exec_()
        
    def on_calibration_mode_toggled(self, checked):
        # ★ 차단 로직 삭제: 자동 모드에서도 언제든 클릭 가능
        if checked:
            print("[INFO] 교정 선원 모드 선택됨")
            # 만약 로봇이 작업 중(자동 모드)이었다면, 깔끔하게 정지 및 초기화시킴
            if self.is_working:
                self.on_stop_button_clicked()
                
            self.ui.inspection_mode_btn.blockSignals(True)
            self.ui.inspection_mode_btn.setChecked(False)
            self.ui.inspection_mode_btn.blockSignals(False)
            self.reset_all_robot_variables()
        else:
            if not self.ui.inspection_mode_btn.isChecked():
                self.ui.calibration_mode_btn.blockSignals(True)
                self.ui.calibration_mode_btn.setChecked(True)
                self.ui.calibration_mode_btn.blockSignals(False)

    def on_inspection_mode_toggled(self, checked):
        # ★ 차단 로직 삭제
        if checked:
            print("[INFO] 시료 비커 모드 선택됨")
            if self.is_working:
                self.on_stop_button_clicked()
                
            self.ui.calibration_mode_btn.blockSignals(True)
            self.ui.calibration_mode_btn.setChecked(False)
            self.ui.calibration_mode_btn.blockSignals(False)
            self.reset_all_robot_variables()
        else:
            if not self.ui.calibration_mode_btn.isChecked():
                self.ui.inspection_mode_btn.blockSignals(True)
                self.ui.inspection_mode_btn.setChecked(True)
                self.ui.inspection_mode_btn.blockSignals(False)

    # def reset_all_robot_variables(self):
    #     try:
    #         config_path = ROBOT_PATH / "robot_var_config.json"
    #         if config_path.exists():
    #             with open(config_path, 'r', encoding='utf-8') as f:
    #                 config_data = json.load(f)
    #                 var_outputs = config_data.get("ROBOT_VAR_OUTPUT", {})
    #                 io_outputs = config_data.get("ROBOT_IO_OUTPUT", {})
    #             for addr_str, var_name in var_outputs.items():
    #                 self.set_variable_with_ui(var_name, 0)
    #             for addr_str, var_name in io_outputs.items():
    #                 self.set_variable_with_ui(var_name, 0)
    #         print("[INFO] 로봇 제어(OUTPUT) 변수 전체 0 초기화")
    #     except Exception as e: print(f"[ERROR] 변수 초기화 실패: {e}")
    
    def reset_all_robot_variables(self):
        try:
            config_path = ROBOT_PATH / "robot_var_config.json"
            if config_path.exists():
                # 리셋에서 제외할 변수 리스트
                exclude_vars = [
                    "jig_center_offs", 
                    "jig_height_offs", 
                    "cell_deep_offs", 
                    "sensing_offs"
                ]
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    var_outputs = config_data.get("ROBOT_VAR_OUTPUT", {})
                    io_outputs = config_data.get("ROBOT_IO_OUTPUT", {})

                # 1. 내부 변수(Register) 초기화 (오프셋 제외)
                for addr_str, var_name in var_outputs.items():
                    if var_name not in exclude_vars:
                        self.set_variable_with_ui(var_name, 0)
                
                # 2. 물리 출력(IO) 초기화
                for addr_str, var_name in io_outputs.items():
                    if var_name not in exclude_vars:
                        self.set_variable_with_ui(var_name, 0)

            print("[INFO] 로봇 제어 변수 초기화 완료 (오프셋 변수 제외)")
        except Exception as e: 
            print(f"[ERROR] 변수 초기화 실패: {e}")
            
    def on_main_btn_clicked(self): self.ui.stackedWidget.setCurrentWidget(self.ui.main_page)
    def on_manage_btn_clicked(self): self.ui.stackedWidget.setCurrentWidget(self.ui.manage_page)
    def on_robot_btn_clicked(self): self.ui.stackedWidget.setCurrentWidget(self.ui.robot_page)
    
    def check_initial_robot_alarm(self):
        try:
            status_res = self.robot_29999.send_command_29999("status")
            if not status_res: return
            safety_mode_str = None
            for line in status_res.split('\n'):
                if "SafetyMode:" in line:
                    safety_mode_str = line.split(":")[1].strip()
                    break
            if safety_mode_str:
                safe_modes = ["NORMAL", "REDUCED", "1", "2"]
                if safety_mode_str.upper() not in safe_modes:
                    self.report_alarm_signal.emit(f"로봇 세이프티 알람 (상태: {safety_mode_str})", "ERROR")
        except Exception: pass

    def check_connection_alive(self):
        try:
            status_res = self.robot_29999.send_command_29999("status")
            if not status_res or "Error" in status_res: return False
            if "RobotMode:" in status_res and "SafetyMode:" in status_res: return True 
            return False 
        except Exception: return False
        
    def show_message(self, title, message, icon=QMessageBox.Information, buttons=QMessageBox.Ok, text_color="#FFFFFF", font_size=22, button_color="#0078D7", button_text_color="#FFFFFF"):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(500)
        dialog.setMinimumHeight(280)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 30, 20, 30)
        lbl_msg = QLabel(message)
        lbl_msg.setAlignment(Qt.AlignCenter)
        lbl_msg.setWordWrap(True)
        lbl_msg.setStyleSheet(f"color: {text_color}; font-size: {font_size}px; font-weight: bold;")
        layout.addWidget(lbl_msg)
        layout.addSpacing(30)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        button_map = [(QMessageBox.Ok, "확인"), (QMessageBox.Yes, "예"), (QMessageBox.No, "아니오"), (QMessageBox.Cancel, "취소"), (QMessageBox.Close, "닫기")]
        
        for flag, text in button_map:
            if buttons & flag:
                btn = QPushButton(text)
                btn.setFixedSize(130, 60) 
                btn.clicked.connect(lambda _, result=flag: dialog.done(result))
                btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.setStyleSheet(f"QDialog {{ background-color: #2b2b2b; border: 2px solid {button_color}; }} QLabel {{ background-color: transparent; }} QPushButton {{ background-color: #555555; color: {button_text_color}; font-size: 18px; font-weight: bold; border-radius: 5px; border: 1px solid #888; }} QPushButton:hover {{ background-color: {button_color}; border: 1px solid #005a9e; }}")

        parent_geo = self.geometry()
        dialog.move(parent_geo.x() + int((parent_geo.width() - dialog.sizeHint().width()) / 2), parent_geo.y() + int((parent_geo.height() - dialog.sizeHint().height()) / 2))
        return dialog.exec_()
    
    def check_auto_mode_restriction(self):
        if getattr(self, 'auto_mode', False):
            self.show_message("조작 불가", "<b>자동(Auto) 모드</b> 실행 중에는<br>설정을 변경할 수 없습니다.<br><br>수동(Manual) 모드로 전환해 주세요.", QMessageBox.Warning)
            return False
        return True
 
    @pyqtSlot(str, str)
    def add_alarm_log(self, message, level="INFO"):
        time_str = QDateTime.currentDateTime().toString("HH:mm:ss")
        
        if level == "ERROR":
            # self.set_variable_with_ui("system_ng", 1)
            
            # ★ [추가] 에러(알람) 발생 시 즉시 작업 정지(Stop) 시퀀스 실행
            if self.is_working or getattr(self, 'is_manual_moving', False):
                print(f"[SYSTEM] 알람 감지: {message} -> 진행 중인 모든 동작 긴급 정지")
                self.on_stop_button_clicked()
        
        # 시스템 로그 UI 영역에 직관적인 색상으로 출력
        if hasattr(self.ui, 'system_msg_area'):
            if level == "ERROR":
                color, prefix = "#FF3333", "🔴"
            elif level == "WARN":
                color, prefix = "#FFC107", "🟡"
            else:
                color, prefix = "#FFFFFF", "⚪"
                
            html_msg = f"<b style='color: {color};'>[{time_str}] {prefix} {message}</b>"
            self.ui.system_msg_area.append(html_msg)

        if not getattr(self, 'alarm_popup_shown', False):
            self.alarm_popup_shown = True
            self.robot_status_update()
            QApplication.processEvents() 
            self.show_alarm_popup(is_error=(level == "ERROR"), msg=message)
                
    @pyqtSlot()
    def show_alarm_popup(self, is_error=True, msg=""):
        if getattr(self, 'active_alarm_dialog', None) is not None: return
        self.active_alarm_dialog = AlarmDialog(self, is_error, msg)
        self.active_alarm_dialog.closed_signal.connect(self.on_alarm_popup_closed)
        self.active_alarm_dialog.show()
        
    def on_alarm_popup_closed(self):
        self.active_alarm_dialog = None
        
    @pyqtSlot()
    def on_robot_disconnected(self):
        self.robot_disconnected = True
        self.robot_status_update()
        
    def on_manual_mode(self, checked):
        if checked: self.auto_mode = False
            
    def is_alarm_state(self) -> bool:
        try:
            s_mode = int(float(self.safety_mode)) if self.safety_mode is not None else 0
            if s_mode in [3, 4, 5, 6, 7, 8, 9, 11, 12, 13]: return True
        except Exception: pass
        return False
    
    def is_idle_state(self):
        try:
            r_mode = int(float(self.robot_mode)) if self.robot_mode is not None else 0
            return self.is_robot_power_on and r_mode == 5
        except Exception: return False
    
    @pyqtSlot()
    def robot_status_update(self):
        self.is_inspection_mode_cached = self.ui.inspection_mode_btn.isChecked()
        self.current_beaker_name_cached = self.ui.current_beaker.text()
        state_values = self.values
        try:
            if state_values is not None:
                if not self.is_alarm_state(): self.alarm_popup_shown = False
                if not self._speed_initialized and getattr(self, 'robot_speed', None) is not None:
                    self._speed_initialized = True
                    self.ui.speed_slider.blockSignals(True)
                    self.ui.speed_slider.setValue(int(self.robot_speed))
                    self.ui.speed_slider.blockSignals(False)
                    self.ui.speed_label.setText(f"{int(self.robot_speed)}%")

                # ★ 1. 작업 시작/정지 버튼 프리징 해결 (상태가 다를 때만 갱신)
                if self.is_working:
                    if self.ui.start_btn.isEnabled(): self.ui.start_btn.setEnabled(False)
                    if not self.ui.stop_btn.isEnabled(): self.ui.stop_btn.setEnabled(True)
                    
                    if self.ui.mode_label.text() != "자동 작업 중":
                        self.ui.mode_circle.setStyleSheet("border-radius: 30px; background-color: #007BFF;")
                        self.ui.mode_label.setText("자동 작업 중")
                else:
                    if not self.ui.start_btn.isEnabled(): self.ui.start_btn.setEnabled(True)
                    if self.ui.stop_btn.isEnabled(): self.ui.stop_btn.setEnabled(False)
                    
                    target_text = "자동 모드" if getattr(self, 'auto_mode', False) else "수동 모드"
                    target_color = "#007BFF" if getattr(self, 'auto_mode', False) else "#28a745"
                    
                    if self.ui.mode_label.text() != target_text:
                        self.ui.mode_circle.setStyleSheet(f"border-radius: 30px; background-color: {target_color};")
                        self.ui.mode_label.setText(target_text)

                # ====================================================================
                # ★ [핵심 수정] UI 로봇 상태 라벨 (연결 끊김 우선순위 부여)
                # ====================================================================
                if getattr(self, 'robot_disconnected', True):
                    # 프로그램이 연결 끊김으로 판단했다면, 캐시된 데이터 무시하고 무조건 회색 렌더링
                    state_color, state_text = "#888888", "연결 끊김"
                else:
                    robot_mode = getattr(self, 'robot_mode', 0)
                    safety_mode = getattr(self, 'safety_mode', 0)
                    state_color, state_text = "gray", "상태 알 수 없음"

                    if safety_mode in [6, 7]: state_color, state_text = "#dc3545", "비상 정지"
                    elif safety_mode in [8, 9]: state_color, state_text = "#dc3545", "안전 위반"
                    elif safety_mode in [3, 5, 12, 13]: state_color, state_text = "#ffc107", "안전 정지"
                    elif safety_mode == 4: state_color, state_text = "#ffc107", "복구 모드"
                    else:
                        if robot_mode == 0: state_color, state_text = "#888888", "연결 끊김"
                        elif robot_mode == 3: state_color, state_text = "#888888", "전원 꺼짐"
                        elif robot_mode in [1, 2, 4, 5, 8, 9]: state_color, state_text = "#ffc107", "부팅 / 준비 중"
                        elif robot_mode == 7: state_color, state_text = "#28a745", "전원 켜짐"
                        elif robot_mode == 6: state_color, state_text = "#6f42c1", "백드라이브"

                self.ui.robot_state_circle.setStyleSheet(f"border-radius: 30px; background-color: {state_color};")
                self.ui.robot_state_label.setText(state_text)
        except Exception: pass

    @pyqtSlot()
    def on_stop_button_clicked(self):
        print("[CMD] 정지 (Stop) 버튼 클릭됨")
        
        # =======================================================
        # ★ 1. UI 및 상태 즉시 변경 (버튼 누르자마자 0.01초 컷 반응)
        # =======================================================
        self.is_working = False
        self.robot_status_update() # UI 라벨 및 버튼 활성화 상태 즉시 갱신
        QApplication.processEvents() # 바뀐 UI 상태를 강제로 즉시 화면에 렌더링!

        # =======================================================
        # ★ 2. 무거운 통신(Modbus) 작업은 백그라운드 스레드로 던짐
        # =======================================================
        def _stop_sequence_thread():
            try:
                if not getattr(self, 'robot_disconnected', True):
                    # 로봇 제어기 정지 명령 (29999) 전송
                    self.hard_reset_all_sequences()
                    self.robot_29999.send_command_29999("stop")
                
                # 모드버스 제어 신호 전체 0 초기화 (이게 가장 오래 걸리는 주범)
                self.reset_all_robot_variables()
                self.is_manual_moving = False 
                
                # 로봇 초기화 시퀀스도 강제로 멈추기 위해 IDLE로 덮어씀
                self.init_state = "IDLE" 
                
                # with self.state_lock:
                #     for area in [1, 2]:
                #         self.cell_data[area]["seq_state"] = "IDLE"
            except Exception as e: 
                print(f"[STOP ERROR] {e}")
            finally:
                self.pause_event.set() # 내부 스레드 락 해제 보장

        # 비서에게 통신 지시
        threading.Thread(target=_stop_sequence_thread, daemon=True).start()


    def on_start_work_btn_clicked(self):
        if getattr(self, 'robot_disconnected', True):
            self.show_message("조작 불가", "로봇이 연결되어 있지 않습니다.", QMessageBox.Warning)
            return
        
        # 2. [추가] 수동 모드 체크 (auto_mode가 False인 경우)
        if not getattr(self, 'auto_mode', False):
            self.show_message("조작 불가", "현재 로봇이 <b>수동 모드</b>입니다.<br>작업을 시작할 수 없습니다.", QMessageBox.Warning)
            # 버튼 상태 원상복구 (혹시 UI에서 눌린 상태로 보일 수 있으므로)
            self.robot_status_update()
            return

        if getattr(self, 'is_manual_moving', False):
            self.show_message("이동 중", "현재 로봇이 이동 중입니다. 완료 후 조작해주세요.", QMessageBox.Warning)
            return

        is_calib = self.ui.calibration_mode_btn.isChecked()
        calib_cell = 0
        
        # 팝업창은 무조건 메인 스레드에서 띄워야 하므로 스레드 분리 전에 실행합니다.
        if is_calib:
            dialog = CellSelectionDialog(self)
            if dialog.exec_() == QDialog.Accepted and dialog.selected_cell in [1, 2]:
                calib_cell = dialog.selected_cell
                self.calib_target_cell = calib_cell
            else:
                return # 취소 누르면 시작 중단

        print("[CMD] 시작 (Start) 버튼 클릭됨")
        
        # =======================================================
        # ★ 1. UI 및 상태 즉시 변경 (버튼 누르자마자 0.01초 컷 반응)
        # =======================================================
        self.is_working = True
        self.is_startup_ready = False
        self.robot_status_update() # 시작 버튼 끄고, 정지 버튼 켜고, 텍스트 파란색 변경
        QApplication.processEvents() # 바뀐 UI 상태를 강제로 즉시 화면에 렌더링!
        
        self.hard_reset_all_sequences()

        # =======================================================
        # ★ 2. 무거운 통신(Modbus) 작업은 백그라운드 스레드로 던짐
        # =======================================================
        def _start_sequence_thread():
            try:
                # 완료 배열 청소
                for area in [1, 2]:
                    cell = self.cell_data[area]
                    if cell["pos"] == 0:
                        cell["done_points"] = []
                        if hasattr(self, 'db_manager'):
                            self.db_manager.update_work_done_points(str(area), [])
                
                # 수십 개의 모드버스 통신 덩어리들 (프리징의 주범들) 
                self.reset_all_robot_variables()
                self.send_product_offsets_to_robot()
                
                # 이전 작업 상태 초기화
                self.init_state = "IDLE"
                with self.state_lock:
                    for area in [1, 2]:
                        self.cell_data[area]["seq_state"] = "IDLE"
                        self.cell_data[area]["last_action"] = "none"
                
                # 29999 소켓 통신 전송
                self.robot_29999.send_command_29999("remote")
                time.sleep(0.1)
                self.robot_29999.send_command_29999("play")
                
                with self.state_lock:
                    for area in [1, 2]:
                        # 로봇이 이미 작업 중 신호를 보내고 있다면?
                        if self.cached_robot_vars.get("input_job_ing", 0) == 1:
                            self.cell_data[area]["seq_state"] = "INPUT_ING"
                            print(f"[RECOVERY] 셀 {area} 로봇 동작 중 감지 -> INPUT_ING 상태로 강제 복구")
                        elif self.cached_robot_vars.get("output_job_ing", 0) == 1:
                            self.cell_data[area]["seq_state"] = "OUTPUT_ING"
                            print(f"[RECOVERY] 셀 {area} 로봇 동작 중 감지 -> OUTPUT_ING 상태로 강제 복구")
                
                # 작업 모드에 따른 초기화 지시
                if is_calib:
                    print(f"[UI] 교정 선원 모드 - 셀 {calib_cell} 작업 시작 (로봇 초기화 후 구동)")
                    # self.calib_target_cell = calib_cell
                    # self.init_state = "REQ_INIT_CALIB" # 교정 모드용 초기화 
                else:
                    print("[UI] 시료 비커 모드 - 로봇 초기화 후 외부 TCP/IP 지시 대기")
                    # self.init_state = "REQ_INIT"
                    
                self.init_state = "REQ_INIT"
                    
            except Exception as e:
                print(f"[START ERROR] 로봇 명령 전송 실패: {e}")

        # 비서에게 통신 지시
        threading.Thread(target=_start_sequence_thread, daemon=True).start()
        
    def hard_reset_all_sequences(self):
        """프로그램의 모든 작업 기억과 로봇의 신호를 완전히 강제 초기화합니다."""
        with self.state_lock:
            for area in [1, 2]:
                cell = self.cell_data[area]
                cell["seq_state"] = "IDLE"
                cell["req_sent"] = False      
                cell["sensor_arrive_time"] = 0 
                
                # ========================================================
                # ★ [핵심 수정] 아래 항목들은 GODO 통신 및 일시정지 복구를 위해 
                # 절대로 지우지 않고 그대로 보존(Keep)합니다!
                # ========================================================
                # cell["last_action"] = "none"   (X 삭제)
                # cell["done_points"] = []       (X 삭제)
                # cell["current_point"] = 1      (X 삭제)
                
                # ② 로봇 PLC 신호 강제 OFF (Modbus 쓰기)
                # 로봇이 '나 작업 끝났어'라고 보낸 신호들을 초기화하기 위해 Req를 0으로 밀어버림
                self.set_variable_with_ui(f"c{area}_input_job_req", 0)
                self.set_variable_with_ui(f"c{area}_output_job_req", 0)
                self.set_variable_with_ui(f"c{area}_sensor_req", 0)
                
            # ③ 글로벌 공용 신호 초기화
            self.set_variable_with_ui("sensor_done", 0)
            self.set_variable_with_ui("home_req", 0)
            self.set_variable_with_ui("system_ng", 0)
            self.init_state = "IDLE"
            
            # 로봇 아웃풋 신호 초기화
            self.set_variable_with_ui("cell_ing", 0)
            self.set_variable_with_ui("cell_sensor_arrived", 0)
            self.set_variable_with_ui("input_job_ing", 0)
            self.set_variable_with_ui("input_job_done", 0)
            self.set_variable_with_ui("output_job_ing", 0)
            self.set_variable_with_ui("output_job_done", 0)
            
            
            
        print("[HARD RESET] 모든 작업 기억과 통신 신호가 초기화되었습니다. 클린 상태로 대기합니다.")

    # def on_buzzer_mute_clicked(self):
    #     print("[UI] 부저 뮤트 버튼 클릭 (현재 알람에 한하여 음소거)")
    #     self.is_buzzer_muted = True

    # def on_alarm_rst_btn_clicked(self):
    #     print("[CMD] 알람 리셋 버튼 클릭됨")
    #     self.is_buzzer_muted = False # 알람 리셋 시 뮤트 상태도 함께 해제
        
    #     # 1. 실린더 및 시스템 내부 릴레이 알람 리셋 펄스
    #     self.reset_system_alarms()
        
    #     # 2. 로봇 컨트롤러(29999) 알람 리셋 프로세스 (비동기 처리)
    #     def _reset_robot():
    #         try:
    #             time.sleep(0.5) # 하드웨어 펄스 안정화 대기
    #             if getattr(self, 'is_robot_power_on', False):
    #                 print("[CMD] 보호정지 해제 및 다이얼로그 닫기 (전원 유지 상태)")
    #                 self.robot_29999.send_command_29999("unlockProtectiveStop")
    #                 self.robot_29999.send_command_29999("closeSafetyDialog")
    #             else:
    #                 print("[CMD] 로봇 전원 OFF 상태: 세이프티 시스템 전체 리셋 (safety -r)")
    #                 self.robot_29999.send_command_29999("safety -r")
    #                 self.robot_29999.send_command_29999("unlockProtectiveStop")
    #                 self.robot_29999.send_command_29999("closeSafetyDialog")
                
    #             self.robot_29999.send_command_29999("popup -c")
    #             print("[RESET] 알람 리셋 로봇 통신 완료")
                
    #             # 0.3초 뒤 UI 업데이트 강제 호출을 메인 스레드에 위임
    #             time.sleep(0.3)
    #             QtCore.QMetaObject.invokeMethod(self, "robot_status_update", Qt.QueuedConnection)
    #         except Exception as e:
    #             print(f"[RESET ERROR] 로봇 통신 중 오류: {e}")
                
    #     threading.Thread(target=_reset_robot, daemon=True).start()
    
    def robot_alarm_reset_button(self):
        curr_time = time.time()
        if curr_time - getattr(self, '_last_reset_click', 0) < 2.0:
            print("[WARN] 알람 리셋 연속 클릭 방지")
            return
        self._last_reset_click = curr_time

        print("[RESET] 전용 알람 리셋 시퀀스 시작")
        
        # 팝업이 떠 있다면 닫기
        if getattr(self, 'active_alarm_dialog', None) is not None:
            self.active_alarm_dialog.close()
            self.active_alarm_dialog = None

        # 1. 모드버스 리셋 펄스 구동 (PLC, 하드웨어 릴레이용)
        self.set_variable_with_ui("DO_Alarm_Reset", 1)
        self.set_variable_with_ui("initialize", 1)

        # 2. UI 프리징을 막기 위해 백그라운드 스레드에서 로봇 복구 및 상태 모니터링 진행
        self.reset_thread = ResetRobotThread(self)
        self.reset_thread.request_update_signal.connect(self.robot_status_update)
        self.reset_thread.finished.connect(self._on_reset_sequence_finished)
        self.reset_thread.start()

    @pyqtSlot()
    def _on_reset_sequence_finished(self):
        print("[RESET] 로봇 통신 및 복구 시퀀스 완벽 종료. UI 알람 플래그 초기화.")
        
        # ★ 스레드가 성공적으로 로봇을 정상화 시킨 '이후'에야 화면 플래그를 지워줍니다.
        self.alarm_popup_shown = False
        self.alarm_manager.active_alarms.clear()
        self.set_variable_with_ui("system_ng", 0)
        self.set_variable_with_ui("DO_Alarm_Reset", 0)
        self.set_variable_with_ui("initialize", 0)

        if hasattr(self, 'reset_thread'):
            self.reset_thread.deleteLater()
            
    def on_robot_power_on_button_clicked(self, pressed):
        if not pressed:
            return

        # ★ 1초 이내 연속 클릭 방지
        curr_time = time.time()
        if curr_time - getattr(self, '_last_power_click', 0) < 1.0:
            self.ui.robot_power_on_btn.blockSignals(True)
            self.ui.robot_power_on_btn.setChecked(False)
            self.ui.robot_power_on_btn.blockSignals(False)
            return
        self._last_power_click = curr_time

        print(f"[INFO] Power sequence start. Current: Power={self.is_robot_power_on}, Mode={self.robot_mode}")
        
        # 이미 켜져 있고 브레이크도 풀린 상태(7)면 무시
        if self.is_robot_power_on and self.robot_mode == 7:
            print("[INFO] 로봇이 이미 Ready(Running) 상태입니다.")
            self.ui.robot_power_on_btn.blockSignals(True)
            self.ui.robot_power_on_btn.setChecked(False)
            self.ui.robot_power_on_btn.blockSignals(False)
            return

        # ★ 타이머 객체 재사용 로직으로 교체 (프로그램 튕김 방지)
        if not hasattr(self, 'power_on_timer'):
            self.power_on_timer = QtCore.QTimer(self)
            self.power_on_timer.timeout.connect(self.robot_power_on)
        else:
            self.power_on_timer.stop()

        # 딜레이 및 카운터 변수 초기화
        self.retry_counter = 0 

        # 시작 단계 설정
        if self.is_robot_power_on and self.robot_mode == 5:
            print("[INFO] Idle 상태 감지 → 브레이크 해제 단계(STEP 3)부터 시작")
            self.power_on_step = 3
        else:
            print("[INFO] 전원 OFF 상태 감지 → 전체 시퀀스(STEP 0) 시작")
            self.power_on_step = 0

        # 타이머 시작 (0.2초 간격)
        self.power_on_timer.start(200)
        
    def robot_power_on(self):
        try:
            # 로컬 변수로 상태 매핑
            power = self.is_robot_power_on
            mode = self.robot_mode

            # =========================================================
            # STEP 0: 제어권 요청 및 전원 ON 명령
            # =========================================================
            if self.power_on_step == 0:
                self.robot_29999.robot_remote_mode_on()
                self.robot_29999.robot_power_on()
                print("[STEP 0] Remote 모드 전환 및 Power ON 명령 전송")
                self.power_on_step = 1
                return

            # =========================================================
            # STEP 1: 전원 켜기
            # =========================================================
            if self.power_on_step == 1:
                # 전원이 꺼져있거나 상태를 못 읽었으면 지속적으로 명령 전송
                if not power: 
                    self.robot_29999.robot_power_on()
                else:
                    print(f"[STEP 1] Power ON 확인됨 (Power: {power}) → Idle 대기 진입")
                    self.power_on_step = 2
                    self.retry_counter = 0
                return

            # =========================================================
            # STEP 2: Idle 모드 대기
            # =========================================================
            if self.power_on_step == 2:
                # Idle 상태 (5) 확인
                if mode == 5:
                    print(f"[STEP 2] Idle 모드 진입 확인 (Mode: {mode}) → 브레이크 해제 진입")
                    self.power_on_step = 3
                    self.retry_counter = 0
                else:
                    # 로그 폭주 방지: 10틱(2초)마다 한 번씩만 출력
                    if self.retry_counter % 10 == 0:
                        print(f"[STEP 2] Idle 모드 대기 중... 현재 Mode: {mode}")
                    self.retry_counter += 1
                return

            # =========================================================
            # STEP 3: 브레이크 해제
            # =========================================================
            if self.power_on_step == 3:
                # 1. Ready(7)가 되었다면 성공
                if mode == 7:
                    print("[STEP 3] 로봇 Ready 상태 변경 확인! → 완료 단계로 이동")
                    self.power_on_step = 4
                    return

                # 2. 아직 7이 아니라면 5틱(1초)마다 한 번씩 브레이크 해제 반복 전송
                if self.retry_counter % 5 == 0:
                    self.robot_29999.robot_brakeRelease()
                    print(f"[STEP 3] 브레이크 해제 명령 전송... (현재 Mode: {mode})")
                
                self.retry_counter += 1
                
                # 75틱(15초) 이상 무한 대기 시 에러 간주
                if self.retry_counter > 75:
                    raise Exception("브레이크 해제 대기 시간 초과 (15초)")
                return

            # =========================================================
            # STEP 4: 최종 완료 처리
            # =========================================================
            if self.power_on_step == 4:
                print("[STEP 4] 로봇 부팅 시퀀스 완료.")
                self.power_on_timer.stop()

                # 토글 버튼이 눌린 채로 남아있지 않게 원복 (UI 시각화)
                self.ui.robot_power_on_btn.blockSignals(True)
                self.ui.robot_power_on_btn.setChecked(False)
                self.ui.robot_power_on_btn.blockSignals(False)
                return

        except Exception as e:
            print(f"[ERROR] Power ON 시퀀스 오류: {e}")
            import traceback
            traceback.print_exc()
            
            self.power_on_timer.stop()
            self.ui.robot_power_on_btn.blockSignals(True)
            self.ui.robot_power_on_btn.setChecked(False)
            self.ui.robot_power_on_btn.blockSignals(False)

    def on_home_button_clicked(self):
        curr_time = time.time()
        if curr_time - getattr(self, '_last_home_click_time', 0) < 1.0: return
        self._last_home_click_time = curr_time

        try:
            print("[INFO] 로봇 홈 이동 요청")
            if getattr(self, 'is_manual_moving', False):
                self.show_message("조작 불가", "현재 이동 중입니다.<br>동작이 끝난 후 눌러주세요.", QMessageBox.Warning)
                return
                
            self.is_manual_moving = True 
            
            # =================================================================
            # ★ 핵심 안전장치: 홈으로 이동하기 전에 기존에 쏘아둔 타겟 이동 명령들을 확실하게 끕니다.
            # =================================================================
            manual_reqs = [
                "c1_sensor_req", "c2_sensor_req",
                "c1_manl_move_req", "c2_manl_move_req",
                "j1_manl_targ_req", "j2_manl_targ_req",
                "grip_open", "grip_close"
            ]
            for req in manual_reqs:
                self.set_variable_with_ui(req, 0)
                
            # 기존 전체 변수 초기화 로직 (2중 방어망)
            self.reset_all_robot_variables()
            
            # 로봇 동작 초기화 (기존 동작 취소)
            self.robot_29999.robot_stop()
            
            def start_and_go_home():
                if getattr(self, 'robot_disconnected', True):
                    self.is_manual_moving = False
                    return
                self.robot_29999.robot_play()
                # 0.5초 대기 후 홈 이동 명령(home_req=1) 하달
                QtCore.QTimer.singleShot(500, self._execute_home_move)

            # 로봇 플레이 전 0.5초 딜레이 (신호가 0으로 확실히 떨어질 시간 확보)
            QtCore.QTimer.singleShot(500, start_and_go_home)
            
        except Exception as e:
            print(f"[ERROR] 홈 이동 초기화 실패: {e}")
            self.is_manual_moving = False

    def _execute_home_move(self):
        try:
            self.set_variable_with_ui("home_req", 1)

            if hasattr(self, 'home_check_timer') and self.home_check_timer.isActive():
                self.home_check_timer.stop()
                
            raw_home = self.robot_29999.get_variable("J_home")
            self._target_home_joints = ast.literal_eval(raw_home) if isinstance(raw_home, str) else raw_home
                
            self.home_check_timer = QtCore.QTimer(self)
            self.home_check_timer.timeout.connect(self.check_home_arrival)
            self.home_timeout_count = 0
            self.home_check_timer.start(100) 
        except Exception:
            self.is_manual_moving = False

    def check_home_arrival(self):
        try:
            self.home_timeout_count += 1
            is_home_val = self.get_variable_with_ui("is_home")
            is_home_true = (str(is_home_val).lower() in ["true", "1"]) or (is_home_val == 1) or (is_home_val is True)
            
            near_home = False
            if hasattr(self, '_target_home_joints') and isinstance(self._target_home_joints, list) and len(self._target_home_joints) == 6:
                cur_j = self.get_current_joints()
                dist = sum(abs(c - h) for c, h in zip(cur_j, self._target_home_joints))
                if dist < 0.05: near_home = True
            
            if is_home_true or near_home:
                self.set_variable_with_ui("home_req", 0)
                self.home_check_timer.stop()
                if hasattr(self.ui, 'home_pose_btn'):
                    self.ui.home_pose_btn.blockSignals(True)
                    self.ui.home_pose_btn.setChecked(False)
                    self.ui.home_pose_btn.blockSignals(False)
                self.is_manual_moving = False 
                
                self.apply_mode_ui_restrictions() 
                return

            if self.home_timeout_count > 300: 
                self.set_variable_with_ui("home_req", 0)
                self.home_check_timer.stop()
                if not getattr(self, 'auto_mode', False): self.robot_29999.robot_stop()
                self.is_manual_moving = False
        except Exception:
            self.home_check_timer.stop()
            self.is_manual_moving = False

    def on_slider_changed(self, value): self.ui.speed_label.setText(f"{value}%")
    
    def on_speed_changed(self, value):
        self.ui.speed_label.setText(f"{value}%")
        self.robot_29999.set_robot_speed(speed=value)

    def init_robot_var_ui(self, font_size=14):
        try:
            table = self.ui.robot_var_table 
            table.setRowCount(0) 
            table.setColumnCount(2) 
            table.setHorizontalHeaderLabels(["변수 명", "변수 값"])
            header = table.horizontalHeader()
            header.setStretchLastSection(False) 
            header.setSectionResizeMode(QHeaderView.Stretch) 
            self.var_item_map = {} 

            config_path = ROBOT_PATH / "robot_var_config.json"
            
            var_inputs, var_outputs, io_inputs, io_outputs = {}, {}, {}, {}
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    var_inputs = config_data.get("ROBOT_VAR_INPUT", {})
                    var_outputs = config_data.get("ROBOT_VAR_OUTPUT", {})
                    io_inputs = config_data.get("ROBOT_IO_INPUT", {})
                    io_outputs = config_data.get("ROBOT_IO_OUTPUT", {})

            input_names = list(dict.fromkeys(var_inputs.values()))   
            output_names = list(dict.fromkeys(var_outputs.values())) 
            io_in_names = list(dict.fromkeys(io_inputs.values()))
            io_out_names = list(dict.fromkeys(io_outputs.values()))
            
            base_font = QFont("Arial", font_size)
            bold_font = QFont("Arial", font_size, QFont.Bold)
            row_height = font_size + 20

            def add_section_header(title, bg_color):
                row_idx = table.rowCount()
                table.insertRow(row_idx)
                table.setRowHeight(row_idx, row_height + 10)
                item = QTableWidgetItem(title)
                item.setFont(bold_font)
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QColor(bg_color))
                item.setForeground(QColor("white"))
                item.setFlags(Qt.ItemIsEnabled) 
                table.setItem(row_idx, 0, item)
                table.setSpan(row_idx, 0, 1, 2) 

            def add_vars(var_list, name_bg_color):
                name_bg = QColor(name_bg_color)
                for var_name in var_list:
                    # 연결 끊김 상태면 값은 '-' 로 표시
                    val_text = "-"
                    if not getattr(self, 'robot_disconnected', True):
                        try:
                            val = self.get_variable_with_ui(var_name)
                            if val is not None:
                                if isinstance(val, float): val_text = f"{val:.4f}"
                                else: val_text = str(val)
                        except Exception: pass

                    row_idx = table.rowCount()
                    table.insertRow(row_idx)
                    table.setRowHeight(row_idx, row_height)
                    
                    name_item = QTableWidgetItem(var_name)
                    name_item.setFont(base_font)
                    name_item.setTextAlignment(Qt.AlignCenter) 
                    name_item.setFlags(name_item.flags() ^ Qt.ItemIsEditable) 
                    name_item.setBackground(name_bg) 
                    table.setItem(row_idx, 0, name_item)
                    
                    val_item = QTableWidgetItem(val_text)
                    val_item.setFont(base_font)
                    val_item.setTextAlignment(Qt.AlignCenter) 
                    val_item.setForeground(QBrush(QColor(0, 0, 0))) 
                    val_item.setData(Qt.UserRole, var_name)
                    table.setItem(row_idx, 1, val_item)
                    self.var_item_map[var_name] = val_item

            if input_names:
                add_section_header("▼ 내부 상태 모니터링 (Registers)", "#537491")
                add_vars(input_names, "#537491")
            if output_names:
                add_section_header("▼ 내부 제어 명령 (Registers)", "#28a745")
                add_vars(output_names, "#28a745")
            if io_in_names:
                add_section_header("▼ 물리 입력 IO (Coils)", "#17a2b8")
                add_vars(io_in_names, "#17a2b8")
            if io_out_names:
                add_section_header("▼ 물리 출력 IO (Coils)", "#d39e00")
                add_vars(io_out_names, "#d39e00")

            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            try: table.cellDoubleClicked.disconnect()
            except: pass
            table.cellDoubleClicked.connect(self.on_robot_var_table_double_click)
            
        except Exception as e: 
            print(f"[UI INIT ERROR] {e}")
            
    def on_robot_var_table_double_click(self, row, col):
        if not self.check_auto_mode_restriction():
            self.ui.robot_var_table.clearFocus() 
            return
        if col != 1: return
        item = self.ui.robot_var_table.item(row, col)
        if not item: return
        var_key = item.data(Qt.UserRole)
        old_value = item.text()
        
        new_value, ok = self.show_custom_input_dialog(
            title=f"변수 값 수정 ({var_key})", 
            label_text="새로운 값을 입력하세요:", 
            echo_mode=QLineEdit.Normal, 
            default_value=old_value
        )
        self.hide_touch_keyboard()
        
        if ok and new_value:
            try:
                if "." in new_value: final_val = float(new_value)
                else: final_val = int(new_value)
            except ValueError: final_val = new_value 
            self.set_variable_with_ui(var_key, final_val)
        self.ui.robot_var_table.clearFocus()
        self.setFocus()
        
    def _load_modbus_config(self):
        config_path = ROBOT_PATH / "robot_var_config.json"
        self.var_inputs = {}
        self.var_outputs = {}
        self.io_inputs = {}
        self.io_outputs = {}
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self.var_inputs = config_data.get("ROBOT_VAR_INPUT", {})
                    self.var_outputs = config_data.get("ROBOT_VAR_OUTPUT", {})
                    self.io_inputs = config_data.get("ROBOT_IO_INPUT", {})
                    self.io_outputs = config_data.get("ROBOT_IO_OUTPUT", {})
            except Exception as e:
                print(f"[CONFIG ERROR] 로봇 IO/변수 설정 로드 실패: {e}")

    def get_variable_with_ui(self, var_name):
        value = self.cached_robot_vars.get(var_name, 0)
        self.update_robot_var_ui(var_name, value)
        return value

    def set_variable_with_ui(self, var_name, value):
        try:
            if not hasattr(self, 'var_outputs'): self._load_modbus_config()
            
            target_reg_addr = None
            target_coil_addr = None
            
            # 1. Output Register에 속하는지 확인
            for addr_str, name in self.var_outputs.items():
                if name == var_name:
                    target_reg_addr = int(addr_str)
                    break
            
            # 2. Output Register에 없으면 Output IO(Coil)에 속하는지 확인
            if target_reg_addr is None:
                for addr_str, name in self.io_outputs.items():
                    if name == var_name:
                        target_coil_addr = int(addr_str)
                        break

            # 값 정제
            str_val = str(value).strip().lower()
            if str_val in ['true', '1']: send_val = 1
            elif str_val in ['false', '0', 'none', '']: send_val = 0
            else:
                try: send_val = int(float(str_val))
                except ValueError: send_val = 0
                
            if not getattr(self, 'robot_disconnected', True) and getattr(self, 'modbus_client', None) is not None:
                with self.modbus_lock:
                    # 레지스터 쓰기
                    if target_reg_addr is not None:
                        if hasattr(self.modbus_client, 'set_register'):
                            self.modbus_client.set_register(target_reg_addr, send_val)
                    
                    # 물리 IO (코일) 쓰기
                    elif target_coil_addr is not None:
                        if hasattr(self.modbus_client, 'write_coil'):
                            self.modbus_client.write_coil(target_coil_addr, bool(send_val))
                        elif hasattr(self.modbus_client, 'set_coil'):
                            self.modbus_client.set_coil(target_coil_addr, bool(send_val))

            self.cached_robot_vars[var_name] = send_val
            self.update_robot_var_ui(var_name, send_val)
            return True
        except Exception as e:
            print(f"[MODBUS WRITE ERROR] {e}")
            return False
     
    def show_touch_keyboard(self):
        if platform.system() == "Windows":
            try: subprocess.Popen("osk", shell=True)
            except Exception: pass

    def hide_touch_keyboard(self):
        if hasattr(self.ui, 'ip_table'):
            self.ui.ip_table.clearSelection()  
            self.ui.ip_table.clearFocus()      
            self.ui.ip_table.setCurrentItem(None) 
        if hasattr(self.ui, 'robot_var_table'):
            self.ui.robot_var_table.clearSelection()
            self.ui.robot_var_table.clearFocus()
            self.ui.robot_var_table.setCurrentItem(None)
        if hasattr(self.ui, 'centralwidget'): self.ui.centralwidget.setFocus()
        else: self.setFocus()
            
        if platform.system() == "Windows":
            try:
                hwnd = ctypes.windll.user32.FindWindowW(u"OSKMainClass", None)
                if hwnd: ctypes.windll.user32.PostMessageW(hwnd, 0x0112, 0xF060, 0)
            except Exception: pass
            try:
                subprocess.call("taskkill /IM osk.exe /F", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                subprocess.call("taskkill /IM TabTip.exe /F", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except Exception: pass
            
    def show_custom_input_dialog(self, title, label_text, echo_mode=QLineEdit.Normal, default_value=""):
        dialog = QInputDialog(self)
        dialog.setWindowTitle(title)
        dialog.setLabelText(label_text)
        dialog.setTextValue(default_value)
        dialog.setTextEchoMode(echo_mode)
        dialog.setStyleSheet("""
            QInputDialog { background-color: #2b2b2b; border: 2px solid #0078D7; }
            QLabel { color: white; font-size: 20px; font-weight: bold; background-color: #2b2b2b; margin-bottom: 10px; }
            QLineEdit { background-color: white; color: black; font-size: 24px; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
            QPushButton { background-color: #555555; color: white; font-size: 18px; font-weight: bold; border-radius: 5px; min-width: 120px; min-height: 50px; margin: 5px; }
            QPushButton:hover { background-color: #0078D7; }
        """)
        dialog.setFixedSize(500, 280)
        parent_geo = self.geometry()
        dialog.move(parent_geo.x() + int((parent_geo.width() - dialog.sizeHint().width()) / 2), parent_geo.y() + int((parent_geo.height() - dialog.sizeHint().height()) / 2))
        result = dialog.exec_()
        self.hide_touch_keyboard()
        self.setFocus()
        return dialog.textValue(), (result == QDialog.Accepted)
    
    # =========================================================================
    # [수동 모드 제어] 모드 선택, 셀/지그 타겟 이동, 그리퍼 제어
    # =========================================================================
    def init_manual_control_buttons(self):
        try: self.ui.home_pose_btn.clicked.disconnect()
        except: pass
        self.ui.home_pose_btn.clicked.connect(self.on_home_button_clicked)

        # ★ [핵심 1] 모든 수동 버튼을 토글(Checkable)로 설정
        self.ui.gripper_mode_btn.setCheckable(True)
        self.ui.sensor_mode_btn.setCheckable(True)
        
        try:
            self.ui.gripper_mode_btn.toggled.disconnect()
            self.ui.sensor_mode_btn.toggled.disconnect()
        except: pass
        
        # ★ [핵심 2] clicked 대신 toggled를 써야 CSS(:checked)가 완벽히 반응합니다.
        self.ui.gripper_mode_btn.toggled.connect(self.on_gripper_mode_toggled)
        self.ui.sensor_mode_btn.toggled.connect(self.on_sensor_mode_toggled)

        self.pose_btns = [
            "cell_1_pose_btn", "cell_2_pose_btn",
            "jig1_point_btn_1", "jig1_point_btn_2", "jig1_point_btn_3", "jig1_point_btn_4",
            "jig1_point_btn_5", "jig1_point_btn_6", "jig1_point_btn_7", "jig1_point_btn_8",
            "jig1_point_btn_9", "jig1_point_btn_10", "jig1_point_btn_11", "jig1_point_btn_12",
            "jig2_point_btn_1", "jig2_point_btn_2", "jig2_point_btn_3", "jig2_point_btn_4",
            "jig2_point_btn_5", "jig2_point_btn_6", "jig2_point_btn_7", "jig2_point_btn_8",
            "jig2_point_btn_9", "jig2_point_btn_10", "jig2_point_btn_11", "jig2_point_btn_12"
        ]
        
        for btn_name in self.pose_btns:
            btn_widget = getattr(self.ui, btn_name, None)
            if btn_widget:
                try: btn_widget.toggled.disconnect()
                except: pass
                btn_widget.setCheckable(True)
                btn_widget.setEnabled(False) 
                # 람다 함수에 checked 인자를 정확히 매핑
                btn_widget.toggled.connect(lambda checked, name=btn_name: self.on_pose_btn_toggled(name, checked))
                
        # 그리퍼 제어 (일반 클릭 버튼)
        if hasattr(self.ui, 'gripper_open_btn'):
            self.ui.gripper_open_btn.clicked.connect(lambda: self.on_gripper_control("grip_open"))
            self.ui.gripper_open_btn.setEnabled(False)
        if hasattr(self.ui, 'gripper_close_btn'):
            self.ui.gripper_close_btn.clicked.connect(lambda: self.on_gripper_control("grip_close"))
            self.ui.gripper_close_btn.setEnabled(False)

    def on_gripper_mode_toggled(self, checked):
        # 1. 끌 때 (체크 해제 시)
        if not checked:
            print("[UI] 그리퍼 모드 해제")
            self.lock_all_pose_buttons()
            return

        # 2. 켤 때 (조건 검사)
        if getattr(self, 'robot_disconnected', True) or not self.check_auto_mode_restriction():
            self.ui.gripper_mode_btn.blockSignals(True)
            self.ui.gripper_mode_btn.setChecked(False)
            self.ui.gripper_mode_btn.blockSignals(False)
            return

        is_home_val = self.get_variable_with_ui("is_home")
        is_home_true = str(is_home_val).lower() in ["true", "1"] or is_home_val is True or is_home_val == 1
        if not is_home_true:
            self.show_message("조작 불가", "로봇이 <b>HOME 위치</b>에 있어야 합니다.", QMessageBox.Warning)
            self.ui.gripper_mode_btn.blockSignals(True)
            self.ui.gripper_mode_btn.setChecked(False)
            self.ui.gripper_mode_btn.blockSignals(False)
            return

        # 3. 정상 활성화 (센서 모드가 켜져있으면 끄기)
        if self.ui.sensor_mode_btn.isChecked():
            self.ui.sensor_mode_btn.blockSignals(True)
            self.ui.sensor_mode_btn.setChecked(False)
            self.ui.sensor_mode_btn.blockSignals(False)

        print("[UI] 그리퍼 모드 활성화")
        self.apply_mode_ui_restrictions()

    def on_sensor_mode_toggled(self, checked):
        # 1. 끌 때 (체크 해제 시)
        if not checked:
            print("[UI] 센서 모드 해제")
            self.lock_all_pose_buttons()
            return

        # 2. 켤 때 (조건 검사)
        if getattr(self, 'robot_disconnected', True) or not self.check_auto_mode_restriction():
            self.ui.sensor_mode_btn.blockSignals(True)
            self.ui.sensor_mode_btn.setChecked(False)
            self.ui.sensor_mode_btn.blockSignals(False)
            return

        is_home_val = self.get_variable_with_ui("is_home")
        is_home_true = str(is_home_val).lower() in ["true", "1"] or is_home_val is True or is_home_val == 1
        if not is_home_true:
            self.show_message("조작 불가", "로봇이 <b>HOME 위치</b>에 있어야 합니다.", QMessageBox.Warning)
            self.ui.sensor_mode_btn.blockSignals(True)
            self.ui.sensor_mode_btn.setChecked(False)
            self.ui.sensor_mode_btn.blockSignals(False)
            return

        # 3. 정상 활성화 (그리퍼 모드가 켜져있으면 끄기)
        if self.ui.gripper_mode_btn.isChecked():
            self.ui.gripper_mode_btn.blockSignals(True)
            self.ui.gripper_mode_btn.setChecked(False)
            self.ui.gripper_mode_btn.blockSignals(False)

        print("[UI] 센서 모드 활성화")
        self.apply_mode_ui_restrictions()

    def apply_mode_ui_restrictions(self):
        is_sensor = self.ui.sensor_mode_btn.isChecked()
        is_gripper = self.ui.gripper_mode_btn.isChecked()
        
        for btn_name in self.pose_btns:
            btn_widget = getattr(self.ui, btn_name, None)
            if not btn_widget: continue
            
            # 모드 스위칭 시 기존 타겟 초기화
            btn_widget.blockSignals(True)
            btn_widget.setChecked(False)
            btn_widget.blockSignals(False)
            
            if is_sensor:
                btn_widget.setEnabled(btn_name in ["cell_1_pose_btn", "cell_2_pose_btn"])
            elif is_gripper:
                btn_widget.setEnabled(True)
                
        if hasattr(self.ui, 'gripper_open_btn'): self.ui.gripper_open_btn.setEnabled(is_gripper)
        if hasattr(self.ui, 'gripper_close_btn'): self.ui.gripper_close_btn.setEnabled(is_gripper)

    def lock_all_pose_buttons(self):
        for btn_name in self.pose_btns:
            btn_widget = getattr(self.ui, btn_name, None)
            if btn_widget: 
                btn_widget.blockSignals(True)
                btn_widget.setChecked(False)
                btn_widget.blockSignals(False)
                btn_widget.setEnabled(False)
                
        if hasattr(self.ui, 'gripper_open_btn'): self.ui.gripper_open_btn.setEnabled(False)
        if hasattr(self.ui, 'gripper_close_btn'): self.ui.gripper_close_btn.setEnabled(False)

    def on_pose_btn_toggled(self, btn_name, checked):
        btn_widget = getattr(self.ui, btn_name, None)
        
        # 1. 취소 (다시 눌러서 껐을 때 -> 홈 복귀)
        if not checked:
            print(f"[CMD] 수동 타겟 취소 ({btn_name}) -> 홈 복귀 실행")
            self.on_home_button_clicked()
            self.apply_mode_ui_restrictions() # 다른 버튼 잠금 해제
            return

        # 2. 이동 (버튼을 켰을 때) - 예외 검사
        if getattr(self, 'robot_disconnected', True) or not self.check_auto_mode_restriction():
            if btn_widget: 
                btn_widget.blockSignals(True); btn_widget.setChecked(False); btn_widget.blockSignals(False)
            return

        is_home_val = self.get_variable_with_ui("is_home")
        is_home_true = str(is_home_val).lower() in ["true", "1"] or is_home_val is True or is_home_val == 1
        if not is_home_true:
            self.show_message("조작 불가", "현재 홈(Home) 위치가 아닙니다.<br>먼저 [홈으로 이동] 해주세요.", QMessageBox.Warning)
            if btn_widget: 
                btn_widget.blockSignals(True); btn_widget.setChecked(False); btn_widget.blockSignals(False)
            return

        # 3. 다른 타겟 버튼들 잠금 (라디오 버튼 효과)
        for other_name in self.pose_btns:
            if other_name != btn_name:
                w = getattr(self.ui, other_name, None)
                if w:
                    w.blockSignals(True); w.setChecked(False); w.setEnabled(False); w.blockSignals(False)

        # 4. ★ [핵심] 모드에 맞는 정확한 신호(Req) 및 타겟(Target) 세팅
        is_sensor = self.ui.sensor_mode_btn.isChecked()
        req_vars = []
        target_addr = None
        target_pt = None

        if btn_name == "cell_1_pose_btn":
            if is_sensor:
                # ★ 센서 모드: 포인트 0번 지정 + Input 요청 + Sensor 요청 콤보!
                target_addr = "J1_target_point"
                target_pt = 0
                req_vars = ["c1_input_job_req", "c1_sensor_req"]
            else:
                # 그리퍼 모드: 일반 셀 이동
                req_vars = ["c1_manl_move_req"]
                
        elif btn_name == "cell_2_pose_btn":
            if is_sensor:
                # ★ 센서 모드: 포인트 0번 지정 + Input 요청 + Sensor 요청 콤보!
                target_addr = "J2_target_point"
                target_pt = 0
                req_vars = ["c2_input_job_req", "c2_sensor_req"]
            else:
                # 그리퍼 모드: 일반 셀 이동
                req_vars = ["c2_manl_move_req"]
                
        elif btn_name.startswith("jig1_point_btn_"):
            target_pt = int(btn_name.split("_")[-1])
            target_addr = "J1_target_point"
            req_vars = ["j1_manl_targ_req"]
            
        elif btn_name.startswith("jig2_point_btn_"):
            target_pt = int(btn_name.split("_")[-1])
            target_addr = "J2_target_point"
            req_vars = ["j2_manl_targ_req"]

        # 5. 백그라운드 펄스(Pulse) 스레드로 안전하게 전송
        def _pulse_move_req():
            try:
                # ① 타겟 포인트 설정 (필요한 경우만)
                if target_addr and target_pt is not None:
                    print(f"[CMD] {target_addr} = {target_pt} 설정")
                    self.set_variable_with_ui(target_addr, target_pt)

                # ② 요청 신호들 ON
                for req in req_vars:
                    print(f"[CMD] {req} 신호 ON")
                    self.set_variable_with_ui(req, 1)
                
                time.sleep(0.1) # 로봇이 신호를 인식할 0.1초 찰나의 시간 확보
                print(f"[CMD] 로봇 Play 명령 전송 ({btn_name})")
                self.robot_29999.robot_play()
                
                time.sleep(0.5) # 로봇이 완전히 출발할 때까지 대기
                
                # ③ 잔류 신호 OFF (로봇의 유령 동작 완벽 차단)
                for req in req_vars:
                    print(f"[CMD] {req} 신호 OFF (안전 펄스 처리)")
                    self.set_variable_with_ui(req, 0)
                    
            except Exception as e:
                print(f"[PULSE ERROR] 통신 스레드 에러: {e}")

        threading.Thread(target=_pulse_move_req, daemon=True).start()

    def on_gripper_control(self, action):
        if getattr(self, 'robot_disconnected', True) or not self.check_auto_mode_restriction(): return
        if not self.ui.gripper_mode_btn.isChecked(): return
        
        def _pulse_gripper():
            self.set_variable_with_ui(action, 1)
            print(f"[CMD] 수동 조작: 그리퍼 {action} 전송 완료")
            time.sleep(0.5)
            self.set_variable_with_ui(action, 0)
            
        threading.Thread(target=_pulse_gripper, daemon=True).start()
        
if __name__ == "__main__":
    def qt_excepthook(exc_type, exc_value, exc_tb):
        print("[UNCAUGHT EXCEPTION]")
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = qt_excepthook
    app = QApplication(sys.argv)
    window = GODO()
    # window.showMaximized()
    window.show()
    sys.exit(app.exec_())