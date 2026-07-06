import sys
import socket
import threading
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGridLayout, QGroupBox, QSpinBox)
from PyQt5.QtCore import QThread, QTimer
from pyModbusTCP.server import ModbusServer

# =====================================================================
# 1. 가상 로봇 Dashboard 서버 (포트 29999)
# =====================================================================
class MockRobotDashboard(QThread):
    def __init__(self, host='0.0.0.0', port=29999):
        super().__init__()
        self.host = host
        self.port = port
        self.running = True

    def run(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(5)
        print(f"[MOCK 29999] 시작됨")

        while self.running:
            try:
                client, addr = server_sock.accept()
                while self.running:
                    data = client.recv(1024)
                    if not data: break
                    req = data.decode('utf-8').strip()
                    
                    if "status" in req.lower():
                        res = "RobotMode: 7\nSafetyMode: 1\n"
                    elif "remotecontrol -status" in req.lower():
                        res = "remoteControl -status: true\n"
                    else:
                        res = "ok\n"
                        
                    client.send(res.encode('utf-8'))
            except Exception: pass
            finally:
                if 'client' in locals(): client.close()

# =====================================================================
# 2. 가상 로봇 Real-time 서버 (포트 30001)
# =====================================================================
class MockRobotRealtime(QThread):
    def __init__(self, host='0.0.0.0', port=30001):
        super().__init__()
        self.host = host
        self.port = port
        self.running = True

    def run(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(5)
        print(f"[MOCK 30001] 시작됨")

        dummy_packet = b'\x00' * 1220 
        while self.running:
            try:
                client, addr = server_sock.accept()
                while self.running:
                    client.send(dummy_packet)
                    time.sleep(0.1)
            except Exception: pass
            finally:
                if 'client' in locals(): client.close()

# =====================================================================
# 3. 가상 Modbus TCP 서버 (포트 502)
# =====================================================================
class MockModbusServer(QThread):
    def __init__(self, host='0.0.0.0', port=502):
        super().__init__()
        self.host = host
        self.port = port
        # no_block=True 로 설정하면 내부적으로 스레드가 돕니다.
        self.server = ModbusServer(host=self.host, port=self.port, no_block=True)
        self.running = True

    def run(self):
        try:
            self.server.start()
            print(f"[MOCK 502] 시작됨")
            while self.running:
                time.sleep(0.5)
        except Exception as e:
            print(f"Modbus 에러: {e}")
        finally:
            self.server.stop()

# =====================================================================
# 4. PyQt5 GUI 제어 패널
# =====================================================================
class MockServerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GODO 드라이 테스트 가상 제어반")
        self.resize(800, 600)
        
        # 서버 시작
        self.t_29999 = MockRobotDashboard()
        self.t_30001 = MockRobotRealtime()
        self.t_502 = MockModbusServer()
        
        # DataBank 인스턴스를 가져와서 사용 (에러 해결 핵심)
        self.databank = self.t_502.server.data_bank
        
        self.t_29999.start()
        self.t_30001.start()
        self.t_502.start()

        self.initUI()
        
        # 모니터링 타이머
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.update_monitor)
        self.monitor_timer.start(500)

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)
        
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 1)

        # ---------------- 좌측 UI (입력 제어) ----------------
        box_io_in = QGroupBox("센서 및 스위치 (ROBOT_IO_INPUT)")
        grid_io_in = QGridLayout()
        self.io_btn_dict = {}
        
        io_inputs = {
            0: "실린더1 전진 완료", 1: "실린더1 전진 확인", 2: "실린더1 후진 완료", 3: "실린더1 후진 확인",
            4: "실린더2 전진 완료", 5: "실린더2 전진 확인", 6: "실린더2 후진 완료", 7: "실린더2 후진 확인",
            10: "실린더1 알람", 11: "실린더2 알람", 15: "자동/수동"
        }
        
        row = 0
        for addr, name in io_inputs.items():
            btn = QPushButton(f"[{addr}] {name} : OFF")
            btn.setCheckable(True)
            btn.setStyleSheet("background-color: lightgray;")
            btn.clicked.connect(lambda checked, a=addr, b=btn: self.toggle_coil(a, b, checked))
            grid_io_in.addWidget(btn, row, 0)
            self.io_btn_dict[addr] = btn
            row += 1
        box_io_in.setLayout(grid_io_in)
        left_layout.addWidget(box_io_in)

        box_reg_in = QGroupBox("로봇 상태 전송 (ROBOT_VAR_INPUT)")
        grid_reg_in = QGridLayout()
        self.reg_spin_dict = {}
        
        reg_inputs = {
            271: "is_home", 272: "cell_ing", 273: "cell_sensor_arrived", 274: "input_job_ing",
            276: "input_job_done", 277: "output_job_ing", 278: "output_job_done", 279: "homing",
            280: "cell_door_state1", 281: "cell_door_state2", 282: "initializing"
        }
        
        row = 0
        for addr, name in reg_inputs.items():
            lbl = QLabel(f"[{addr}] {name}")
            spin = QSpinBox()
            spin.setRange(0, 65535)
            spin.valueChanged.connect(lambda val, a=addr: self.set_register(a, val))
            
            btn_on = QPushButton("Set 1")
            btn_on.clicked.connect(lambda _, s=spin: s.setValue(1))
            btn_off = QPushButton("Set 0")
            btn_off.clicked.connect(lambda _, s=spin: s.setValue(0))
            
            grid_reg_in.addWidget(lbl, row, 0)
            grid_reg_in.addWidget(spin, row, 1)
            grid_reg_in.addWidget(btn_on, row, 2)
            grid_reg_in.addWidget(btn_off, row, 3)
            row += 1
        box_reg_in.setLayout(grid_reg_in)
        left_layout.addWidget(box_reg_in)


        # ---------------- 우측 UI (출력 모니터링) ----------------
        box_io_out = QGroupBox("메인 프로그램 명령 감시 (ROBOT_IO_OUTPUT)")
        grid_io_out = QGridLayout()
        self.io_lbl_dict = {}
        
        io_outputs = {
            32: "실린더1 전진", 33: "실린더1 후진", 34: "실린더2 전진", 35: "실린더2 후진",
            40: "Lamp-R", 41: "Lamp-Y", 42: "Lamp-G", 43: "Buzzer", 45: "DO_Alarm_Reset"
        }
        
        row = 0
        for addr, name in io_outputs.items():
            lbl = QLabel(f"[{addr}] {name}: OFF")
            lbl.setStyleSheet("color: gray;")
            grid_io_out.addWidget(lbl, row, 0)
            self.io_lbl_dict[addr] = lbl
            row += 1
        box_io_out.setLayout(grid_io_out)
        right_layout.addWidget(box_io_out)

        box_reg_out = QGroupBox("메인 프로그램 명령 감시 (ROBOT_VAR_OUTPUT)")
        grid_reg_out = QGridLayout()
        self.reg_lbl_dict = {}
        
        reg_outputs = {
            301: "system_ng", 302: "home_req", 303: "c1_sensor_req", 305: "c1_input_job_req",
            307: "c1_output_job_req", 309: "J1_target_point", 311: "sensor_done", 312: "qr_done",
            313: "jig_center_offs", 317: "initialize", 322: "grip_open", 323: "grip_close"
        }
        
        row = 0
        col = 0
        for addr, name in reg_outputs.items():
            lbl = QLabel(f"[{addr}] {name}: 0")
            grid_reg_out.addWidget(lbl, row, col)
            self.reg_lbl_dict[addr] = lbl
            row += 1
            if row > 6:
                row = 0
                col += 1
        box_reg_out.setLayout(grid_reg_out)
        right_layout.addWidget(box_reg_out)

    # UI 조작 -> Modbus 인스턴스의 databank 메모리 쓰기
    def toggle_coil(self, addr, btn, checked):
        self.databank.set_coils(addr, [checked])
        if checked:
            btn.setText(btn.text().replace("OFF", "ON"))
            btn.setStyleSheet("background-color: lightgreen;")
        else:
            btn.setText(btn.text().replace("ON", "OFF"))
            btn.setStyleSheet("background-color: lightgray;")

    def set_register(self, addr, val):
        self.databank.set_words(addr, [val])

    # Modbus 인스턴스의 databank 메모리 읽기 -> UI 업데이트
    def update_monitor(self):
        # IO Output 모니터링 업데이트
        for addr, lbl in self.io_lbl_dict.items():
            val = self.databank.get_coils(addr, 1)
            if val and val[0]:
                lbl.setText(f"[{addr}] {lbl.text().split('] ')[1].split(':')[0]}: ON")
                lbl.setStyleSheet("color: green; font-weight: bold;")
            else:
                lbl.setText(f"[{addr}] {lbl.text().split('] ')[1].split(':')[0]}: OFF")
                lbl.setStyleSheet("color: gray; font-weight: normal;")
                
        # Register Output 모니터링 업데이트
        for addr, lbl in self.reg_lbl_dict.items():
            val = self.databank.get_words(addr, 1)
            if val:
                lbl.setText(f"[{addr}] {lbl.text().split('] ')[1].split(':')[0]}: {val[0]}")

    def closeEvent(self, event):
        self.t_29999.running = False
        self.t_30001.running = False
        self.t_502.running = False
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MockServerGUI()
    ex.show()
    sys.exit(app.exec_())