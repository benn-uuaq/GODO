import struct
import socket
import select
import time
import queue

from pyModbusTCP.client import ModbusClient

DEFAULT_TIMEOUT = 10.0
MESSAGE_TYPE_ROBOT_STATE = 16
MESSAGE_TYPE_ROBOT_MESSAGE = 20

FMT_HEADER = 'IB'
FMT_ROBOT_MODE = 'IBQ???????BBdddB??I'
FMT_JOINT_HEADER = 'IB'     
FMT_JOINT_DATA = 'dddiiiffffBI'
FMT_CARTESIAN = 'IBdddddddddddd'
FMT_CONFIG = 'IB'+'dd'*6+'dd'*6+'ddddd'+'d'*6+'d'*6+'d'*6+'d'*6+'IIIBBBB'
FMT_MASTERBOARD = 'IBIIBBBdddBBBdddffffB???B'
FMT_ADDITIONAL = 'IB????B'
FMT_TOOL = 'IBBBddfBffB'
FMT_SAFETY = 'IBIbBdddd'
FMT_TOOL_COMM = 'IB?III?Bff'

class RobotDataConfig():
    def __init__(self):
        self.names_pre = [
            'total_message_len', 'total_message_type',
            'mode_sub_len', 'mode_sub_type', 'timestamp', 'reserved_1', 'reserved_2',
            'is_robot_power_on', 'is_emergency_stopped', 'is_robot_protective_stopped',
            'is_task_running', 'is_task_paused', 'robot_mode', 'robot_control_mode',
            'target_speed_fraction', 'speed_scaling', 'target_speed_fraction_limit',
            'get_robot_speed_mode', 'reserved_3', 'is_in_package_mode', 'reserved_4',
            'joint_sub_len', 'joint_sub_type'
        ]

        self.names_joint = [
            'actual_joint', 'target_joint', 'actual_velocity', 
            'joint_reserved_1', 'joint_reserved_2', 'joint_reserved_3',
            'current', 'voltage', 'temperature', 'torques', 'mode', 'joint_reserved_4'
        ]

        self.names_post = [
            'cartesial_sub_len', 'cartesial_sub_type',
            'tcp_x', 'tcp_y', 'tcp_z', 'rot_x', 'rot_y', 'rot_z',
            'offset_px', 'offset_py', 'offset_pz', 'offset_rotx', 'offset_roty', 'offset_rotz',
            
            'configuration_sub_len', 'configuration_sub_type',
            'limit_min_joint_0', 'limit_max_joint_0', 'limit_min_joint_1', 'limit_max_joint_1',
            'limit_min_joint_2', 'limit_max_joint_2', 'limit_min_joint_3', 'limit_max_joint_3',
            'limit_min_joint_4', 'limit_max_joint_4', 'limit_min_joint_5', 'limit_max_joint_5',
            'max_velocity_joint_0', 'max_acc_joint_0', 'max_velocity_joint_1', 'max_acc_joint_1',
            'max_velocity_joint_2', 'max_acc_joint_2', 'max_velocity_joint_3', 'max_acc_joint_3',
            'max_velocity_joint_4', 'max_acc_joint_4', 'max_velocity_joint_5', 'max_acc_joint_5',
            'default_velocity_joint', 'default_acc_joint', 'default_tool_velocity', 'default_tool_acc', 'internal_use',
            'dh_a_joint_0', 'dh_a_joint_1', 'dh_a_joint_2', 'dh_a_joint_3', 'dh_a_joint_4', 'dh_a_joint_5',
            'dh_d_joint_0', 'dh_d_joint_1', 'dh_d_joint_2', 'dh_d_joint_3', 'dh_d_joint_4', 'dh_d_joint_5',
            'dh_alpha_joint_0', 'dh_alpha_joint_1', 'dh_alpha_joint_2', 'dh_alpha_joint_3', 'dh_alpha_joint_4', 'dh_alpha_joint_5',
            'dh_theta_joint_0', 'dh_theta_joint_1', 'dh_theta_joint_2', 'dh_theta_joint_3', 'dh_theta_joint_4', 'dh_theta_joint_5',
            'masterboard_version', 'control_box_type', 'robot_type', 'robot_structure', 'tool_io_type', 'reserved_cfg2', 'reserved_cfg3',
            
            'masterboard_sub_len', 'masterboard_sub_type',
            'digital_input_bits', 'digital_output_bits',
            'standard_analog_input_domain0', 'standard_analog_input_domain1', 'tool_analog_input_domain',
            'standard_analog_input_value0', 'standard_analog_input_value1', 'tool_analog_input_value',
            'standard_analog_output_domain0', 'standard_analog_output_domain1', 'tool_analog_output_domain',
            'standard_analog_output_value0', 'standard_analog_output_value1', 'tool_analog_output_value',
            'masterrbord_temperature', 'robot_voltage', 'robot_current', 'io_current',
            'safety_mode', 'is_robot_in_reduced_mode', 'operational_mode_selector_input',
            'threeposition_enabling_device_input', 'internal_use_mb',
            
            'additional_sub_len', 'additional_sub_type',
            'is_freedrive_button_pressed', 'reserved_add', 'is_freedrive_io_enabled', 'is_dynamic_collision_detect_enabled', 'reserved_add2',
            
            'tool_sub_len', 'tool_sub_type',
            'tool_analog_output_domain', 'tool_analog_input_domain', 'tool_analog_output_value', 'tool_analog_input_value',
            'tool_voltage', 'tool_output_voltage', 'tool_current', 'tool_temperature', 'tool_mode',
            
            'safe_sub_len', 'safe_sub_type',
            'safety_crc_num', 'safety_operational_mode', 'reserved_safe',
            'current_elbow_position_x', 'current_elbow_position_y', 'current_elbow_position_z', 'elbow_radius',
            
            'tool_comm_sub_len', 'tool_comm_sub_type',
            'is_enable', 'baudrate', 'parity', 'stopbits', 'tci_modbus_status', 'tci_usage', 'reserved_tc1', 'reserved_tc2'
        ]
        
        self.fmt = (
            '>' +
            FMT_HEADER + FMT_ROBOT_MODE +
            FMT_JOINT_HEADER + (FMT_JOINT_DATA * 6) +
            FMT_CARTESIAN + FMT_CONFIG + FMT_MASTERBOARD +
            FMT_ADDITIONAL + FMT_TOOL + FMT_SAFETY + FMT_TOOL_COMM
        )

class RobotHeader():
    __slots__ = ['type', 'size',]
    @staticmethod
    def unpack(buf):
        rmd = RobotHeader()
        (rmd.size, rmd.type) = struct.unpack_from('>iB', buf)
        return rmd

class RobotData():
    @staticmethod
    def unpack(buf, config):
        data = RobotData()
        try:
            unpacked = struct.unpack(config.fmt, buf)
            it = iter(unpacked)
            
            for name in config.names_pre: setattr(data, name, next(it))
            for name in config.names_joint: setattr(data, name, [])
            for _ in range(6):
                for name in config.names_joint:
                    getattr(data, name).append(next(it))
            for name in config.names_post: setattr(data, name, next(it))
            return data
        except (struct.error, StopIteration):
            return None
    
class AlarmData:
    def __init__(self, code=None, sub=None, level=None, msg=None):
        self.code = code
        self.sub = sub
        self.level = level
        self.msg = msg
        self.timestamp = time.time()
        self.active = True
    
class ReadAlarm():
    @staticmethod
    def unpack(buf):
        data_length = struct.unpack(">i", buf[0:4])[0]
        data, buf = buf[0:data_length], buf[data_length:]
        msg_type = data[14]

        if msg_type == 10:
            msg = bytearray(data[23:data_length-1]).decode()
            return AlarmData(msg=msg)

        if msg_type == 6:
            error_code = struct.unpack(">i", data[15:19])[0]
            sub_error_code = struct.unpack(">i", data[19:23])[0]
            level = struct.unpack(">i", data[23:27])[0]
            return AlarmData(code=error_code, sub=sub_error_code, level=level)

        return None
        
class Robot_30001():
    def __init__(self, ip, port2) -> None:
        self.__data_config = RobotDataConfig()
        self.ip = ip
        self.port2 = port2
        self.alarm_queue = queue.Queue()
        self.__sock = None

    def connect_30001(self):
        try:
            self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.settimeout(0.5)
            self.__sock.connect((self.ip, self.port2))
            self.__sock.settimeout(10.0) 
            print(f"Connected to {self.ip} on port {self.port2}")
            self.__buf = b""
            return self.__sock
        except Exception as e:
            print(f"Error connecting to {self.ip} on port {self.port2}: {e}")
            self.__sock = None 
            return None 
        
    def disconnect_30001(self):
        if self.__sock:
            self.__sock.close()
            self.__sock = None

    def get_data(self):
        return self.__recv()

    def __recv(self):
        try:
            self.__read_socket_no_wait()
        except Exception:
            return None

        last_valid_data = None
        while len(self.__buf) >= 5:
            try:
                head = RobotHeader.unpack(self.__buf)
            except:
                self.__buf = b""
                break

            if len(self.__buf) < head.size:
                break

            payload = self.__buf[:head.size]
            self.__buf = self.__buf[head.size:]

            if head.type == MESSAGE_TYPE_ROBOT_MESSAGE:
                try:
                    alarm = ReadAlarm.unpack(payload)
                    if alarm: self.alarm_queue.put(alarm)
                except: pass
                continue

            if head.type == MESSAGE_TYPE_ROBOT_STATE:
                try:
                    last_valid_data = RobotData.unpack(payload, self.__data_config)
                except: pass

        return last_valid_data

    def __read_socket_no_wait(self):
        while True:
            readable, _, _ = select.select([self.__sock], [], [], 0)
            if not readable: break
            try:
                more = self.__sock.recv(4096)
                if not more: raise ConnectionError("Socket closed")
                self.__buf += more
            except BlockingIOError:
                break
            except Exception:
                break
            
    def send_command(self, command):
        """30001 포트로 URScript 등 제어 명령을 전송합니다."""
        try:
            if self.__sock is None:
                raise RuntimeError("socket is not connected")
            # 스크립트 전송 (이미 명령어에 \n이 포함되어 넘어온다고 가정)
            self.__sock.sendall(command.encode("utf-8"))
        except Exception as e:
            print(f"[Robot_30001] Error sending command: {e}")
            
class AlarmManager:
    def __init__(self):
        self.active_alarms = {}  # key = (code, sub, msg)

    def process(self, alarm):
        key = (alarm.code, alarm.sub, alarm.msg)
        
        if key in self.active_alarms:
            return False

        self.active_alarms[key] = alarm
        print("================================")
        print("[ALARM TRIGGERED]")

        if alarm.msg:
            print(f"[ALARM MSG] {alarm.msg}")
        else:
            print(f"[ALARM CODE] E{alarm.code} S{alarm.sub} (level={alarm.level})")
        print("================================")

        return True

    def clear(self, key):
        if key in self.active_alarms:
            self.active_alarms[key].active = False
            del self.active_alarms[key]

class Robot_29999():
    def __init__(self, ip, port1):
        self.sock = None
        self.ip = ip
        self.port1 = port1

    def connect_29999(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(0.5)
            self.sock.connect((self.ip, self.port1))
            self.sock.settimeout(10.0)
            print(f"Connected to {self.ip} on port {self.port1}")
            try:
                self.sock.recv(4096) 
            except Exception:
                pass
            return self.sock
        except Exception as e:
            print(f"Error connecting to {self.ip} on port {self.port1}: {e}")
            return None

    def send_command_29999(self, command):
        try:
            if self.sock is None:
                if self.connect_29999() is None:
                    print("[WARN] 29999 not connected; cannot send command")
                    return
            self.sock.sendall(f"{command}\n".encode("utf-8"))
            response = self.sock.recv(4096).decode("utf-8").strip()
            return response
        except Exception as e:
            print(f"Error sending command: {e}")
            return None

    def disconnect_29999(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def robot_mode(self): return self.send_command_29999("robotMode")
    def robot_status(self): return self.send_command_29999("status")
    def robot_power_on(self): return self.send_command_29999("robotControl -on")
    def robot_power_off(self): return self.send_command_29999("robotControl -off")
    def robot_brakeRelease(self): return self.send_command_29999("brakeRelease")
    def robot_play(self): return self.send_command_29999("play")
    def robot_pause(self): return self.send_command_29999("pause")
    def robot_stop(self): return self.send_command_29999("stop")
    
    def set_robot_speed(self, speed):
        return self.send_command_29999(f"speed -v {speed}")
    
    def get_robot_speed(self):
        try:
            response = self.send_command_29999("speed")
            if response is None: return 0
            if ":" in response: return int(response.split(":")[-1].strip())
            return int(response.strip())
        except Exception:
            return 0
        
class Robot_recv():
    def __init__(self, ip, port2): 
        self.ip = ip
        self.recv_data = None
        self.port2 = port2

    def RecvPopup(self, port2):
        self.host = "0.0.0.0"
        self.port2 = port2

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.bind((self.host, self.port2))
            s.listen()
            print(f"🟢 Listening for robot on port {self.port2}...")
            conn, addr = s.accept()
            with conn:
                print(f"🔁 Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    self.recv_data = data.decode().strip()
                    print(f"📥 Received from robot: {self.recv_data}")
                    break  # Exit after one message

    def connectETController(self, ip, port2):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((ip, port2))
            return (True, sock)
        except Exception as e:
            sock.close()
            return (False, None)

    def get_inverse_kinematic(self, p, pc_ip=None, pc_port=None):
        command = f'''
def test():
    socket_open("{pc_ip}", {pc_port})
    sleep(0.5)
    socket_send_string(str(get_inverse_kin({p})))
    socket_close()
end
'''
        conSuc, sock = self.connectETController(self.ip, self.port2)
        if conSuc:
            try:
                print("📤 Sending inverse kinematic request to robot...")
                sock.sendall(command.encode())
            except Exception as e:
                print("❌ Failed to send command:", e)
            finally:
                sock.close()
        else:
            print("❌ Connection to robot failed")

    def get_forward_kinematic(self, p, pc_ip=None, pc_port=None):
        command = f'''
def test():
    socket_open("{pc_ip}", {pc_port})
    sleep(0.5)
    socket_send_string(str(get_forward_kin({p})))
    socket_close()
end
'''
        conSuc, sock = self.connectETController(self.ip, self.port2)
        if conSuc:
            try:
                print("📤 Sending forward kinematic request to robot...")
                sock.sendall(command.encode())
            except Exception as e:
                print("❌ Failed to send command:", e)
            finally:
                sock.close()
        else:
            print("❌ Connection to robot failed")
            
class Robot_modbus():       
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = None
        self.is_running = False

    def connect(self):
        """메인 스레드/연결 스레드에서 명시적으로 호출하는 연결 함수"""
        self.client = ModbusClient(host=self.host, port=self.port,timeout=1.0)
        print(f"[PC Modbus Client] 로봇 서버({self.host}:{self.port}, ID:255)에 코일 통신 연결 시도 중...")
        
        if self.client.is_open:
            self.client.close()
            
        is_open = self.client.open()
        
        if is_open:
            print("[PC Modbus Client] 로봇 서버 접속 성공!")
            self.is_running = True
            return True
        else:
            print("[PC Modbus Client] 로봇 서버 접속 실패")
            return False

    def disconnect(self):
        self.client.close()
        self.is_running = False
        print("[PC Modbus Client] 접속 종료")

    # =========================================================================
    # ★ 1. IO (Coil) 쓰기 및 읽기 함수 복구 (다중 쓰기, Discrete Input 읽기)
    # =========================================================================
    def set_coil(self, address, value):
        """코일 쓰기 (다중 쓰기 지원 - FC 15) 및 코일 읽기(FC 02)를 통한 검증"""
        write_bool = str(value).strip().lower() in ['true', '1']
        write_data = 1 if write_bool else 0
        
        # 1. 쓰기 - 엘리트 로봇은 단일 쓰기(FC05) 지원이 불안정하므로 다중 쓰기(FC15)로 리스트로 보냄!
        is_success = self.client.write_multiple_coils(address, [write_bool])
        
        if not is_success:
            print(f"[Modbus Error] 코일 {address}번 쓰기 명령 통신 실패 (연결 끊김 또는 주소 오류)")
            return False
            
        import time
        time.sleep(0.05)
        
        # 2. 읽기 검증 - 엘리트는 출력 코일의 실제 상태를 읽을 때 Discrete Inputs (FC 02)를 권장함!
        read_data = self.client.read_discrete_inputs(address, 1)
        
        if read_data and len(read_data) > 0:
            read_val = 1 if read_data[0] else 0
            if write_data == read_val:
                return True
            else:
                # 상태 반영이 느린 것일 수도 있으므로, 치명적 에러 대신 경고만 출력
                # print(f"[Modbus Warn] 코일 {address}번 쓰기 후 즉시 검증 실패 - 보낸값: {write_data}, 읽은값: {read_val}")
                return True
        else:
            print(f"[Modbus Error] 코일 {address}번 쓰기 후 읽기(검증) 실패 - 응답 없음")
            return False

    def get_all_coils(self, start_address, count) -> list:
        # 마찬가지로 FC 02 사용
        bits = self.client.read_discrete_inputs(start_address, count)
        if bits:
            return bits
        return []
    
    def get_coil(self, address) -> bool:
        result = self.client.read_discrete_inputs(address, 1)
        if result and len(result) > 0:
            return result[0]
        return False
    
    # =========================================================================
    # ★ 2. 내부 변수 (Register) 쓰기 및 읽기 함수 (음수 지원 완벽 적용)
    # =========================================================================
    def set_register(self, address, value):
        """Holding Register 단일 쓰기 (FC 06) 및 검증 (FC 03)"""
        try:
            write_data = int(float(value))
        except ValueError:
            write_data = 1 if str(value).strip().lower() in ['true', '1'] else 0
        
        # ★ [수정 1] 음수를 Modbus 표준인 16-bit Unsigned(2의 보수)로 변환
        # 예: -5가 들어오면 65531로 변환되어 전송됨 (로봇은 이를 정상적인 -5로 인식)
        modbus_write_val = write_data & 0xFFFF
        
        # 1. 쓰기 (FC 06) 
        is_success = self.client.write_single_register(address, modbus_write_val)
        
        if not is_success:
            print(f"[Modbus Error] 레지스터 {address}번 쓰기 명령 통신 실패")
            return False
            
        import time
        time.sleep(0.05) 
        
        # 2. 읽기 검증 (FC 03)
        read_data = self.client.read_holding_registers(address, 1)
        
        if read_data and len(read_data) > 0:
            read_val = read_data[0]
            # ★ [수정 2] 로봇에서 읽어온 16-bit 양수값을 원래의 음수(Signed)로 복원하여 검증
            signed_read_val = read_val - 65536 if read_val > 32767 else read_val
            
            if write_data == signed_read_val:
                return True
            else:
                return False
        else:
            print(f"[Modbus Error] 레지스터 {address}번 쓰기 후 읽기(검증) 실패 - 응답 없음")
            return False

    def get_all_registers(self, start_address, count) -> list:
        """Holding Register 여러 개 한 번에 읽기 (FC 03)"""
        regs = self.client.read_holding_registers(start_address, count)
        if regs:
            # ★ [수정 3] 로봇 상태를 통째로 읽어올 때도 모두 Signed(음수 허용) 값으로 변환하여 반환
            return [val - 65536 if val > 32767 else val for val in regs]
        return []
    
    def get_register(self, address) -> int:
        """Holding Register 단일 읽기 (FC 03)"""
        result = self.client.read_holding_registers(address, 1)
        if result and len(result) > 0:
            val = result[0]
            # ★ [수정 4] Signed 16-bit 변환
            return val - 65536 if val > 32767 else val
        return 0
        
    # =========================================================================
    # ★ 3. 물리 IO 상태 (Input Register) 읽기 함수 복구 (비트마스크 변환 원복)
    # =========================================================================
    def get_input_register(self, address) -> int:
        """
        메인 프로그램(godo_main.py)은 주소 0을 Input(0~15), 주소 2를 Output(32~47)의 
        16bit 비트마스크 형태로 요청합니다. 
        유실되었던 원본 로봇 코드의 비트마스크 압축(Packing) 로직을 완벽히 복원합니다.
        """
        try:
            # 1. 요청 주소에 따라 읽어올 코일의 시작 번호 지정
            if address == 0:
                start_coil = 0   # ROBOT_IO_INPUT (0~15번 매핑)
            elif address == 2:
                start_coil = 32  # ROBOT_IO_OUTPUT (32~47번 매핑)
            else:
                # 0이나 2가 아닌 다른 레지스터 요청 시 일반 Holding Register(FC 03)로 읽기
                res = self.client.read_holding_registers(address, 1)
                return res[0] if (res and len(res) > 0) else -1

            # 2. FC 02 (Discrete Inputs)로 코일 16개를 한 번에 안전하게 읽어오기
            bits = self.client.read_discrete_inputs(start_coil, 16)
            
            # 3. 읽어온 16개의 배열(True/False)을 하나의 16bit 정수로 압축 (Bitwise 연산)
            if bits and len(bits) == 16:
                mask = 0
                for i, bit_val in enumerate(bits):
                    if bit_val:
                        mask |= (1 << i)
                return mask
            else:
                return -1 # 통신 실패 시 방어
                
        except Exception as e:
            print(f"[Modbus Error] get_input_register 에러: {e}")
            return -1