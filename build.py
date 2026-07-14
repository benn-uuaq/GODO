import os
import sys
import subprocess

# ==============================================================================
# 1. 자동 생성할 SPEC 파일 내용 (소스코드 노출 방지 및 리소스 포함 로직)
# ==============================================================================
SPEC_CONTENT = """# -*- mode: python ; coding: utf-8 -*-
import os
import glob
import site
from PyInstaller.utils.hooks import collect_all

# 1. OpenCV는 기본 훅(collect_all)으로 정상 수집됨
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all('cv2')

# =====================================================================
# ★ [완벽 해결] Orbbec SDK 폴더 구조 유지 및 필수 XML 자동 수집
# =====================================================================
orbbec_custom_binaries = []
orbbec_custom_datas = []

# 1) Program Files 내부의 OrbbecSDK 설치 폴더 탐색 (사진에서 확인된 경로)
orbbec_paths = glob.glob(r"C:\\Program Files\\OrbbecSDK*") + glob.glob(r"C:\\Program Files (x86)\\OrbbecSDK*")

for base_path in orbbec_paths:
    bin_dir = os.path.join(base_path, 'bin')
    if os.path.exists(bin_dir):
        # bin 폴더 전체를 재귀적으로 탐색하여 하위 폴더 구조(extensions 등) 그대로 복제
        for root, dirs, files in os.walk(bin_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, bin_dir)
                dest_dir = '.' if rel_dir == '.' else rel_dir # 폴더 구조 복제 핵심
                
                # DLL 및 PYD는 바이너리로
                if file.lower().endswith(('.dll', '.pyd')):
                    if (full_path, dest_dir) not in orbbec_custom_binaries:
                        orbbec_custom_binaries.append((full_path, dest_dir))
                # XML 설정 파일은 데이터로
                elif file.lower().endswith(('.xml', '.md', '.json')):
                    if (full_path, dest_dir) not in orbbec_custom_datas:
                        orbbec_custom_datas.append((full_path, dest_dir))

# 2) 파이썬 모듈 (pyorbbecsdk) 탐색
site_packages = site.getsitepackages()
if hasattr(site, 'getusersitepackages'):
    site_packages.append(site.getusersitepackages())

for sp in site_packages:
    target_dir = os.path.join(sp, 'pyorbbecsdk')
    if os.path.exists(target_dir):
        for f in glob.glob(os.path.join(target_dir, '*.pyd')) + glob.glob(os.path.join(target_dir, '*.dll')):
            if (f, '.') not in orbbec_custom_binaries:
                orbbec_custom_binaries.append((f, '.'))

# DB 폴더 등 필요한 데이터 파일들을 재귀적으로 수집하는 함수
def get_data_files(src_dir, dest_dir):
    files = []
    if not os.path.exists(src_dir):
        return files
        
    for root, dirs, filenames in os.walk(src_dir):
        for filename in filenames:
            # 소스코드(.py, .pyc, .ui) 및 캐시 폴더는 배포에서 제외 (보안)
            if not filename.endswith('.py') and not filename.endswith('.pyc') and not filename.endswith('.ui') and '__pycache__' not in root:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, src_dir)
                if relative_path == '.':
                    target_dir = dest_dir
                else:
                    target_dir = os.path.join(dest_dir, relative_path)
                
                files.append((full_path, target_dir))
    return files

block_cipher = None

# 데이터 파일 합치기
my_datas = cv2_datas + orbbec_custom_datas
my_datas += get_data_files('DB', 'DB')

if os.path.exists('logo.ico'):
    my_datas.append(('logo.ico', '.'))

if os.path.exists('logo.png'):
    my_datas.append(('logo.png', '.'))

# 바이너리와 히든임포트 합치기
binaries = cv2_binaries + orbbec_custom_binaries
hiddenimports = ['serial', 'pyserial', 'PyQt5', 'numpy', 'zxingcpp'] + cv2_hiddenimports

a = Analysis(
    ['godo_main.py'],
    pathex=[],
    binaries=binaries,
    datas=my_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'PySide2', 'PyQt6'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GODO_SYSTEM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, 
    icon='logo.ico' if os.path.exists('logo.ico') else None, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GODO_SYSTEM',
)
"""

def main():
    spec_filename = "GODO_SYSTEM.spec"
    
    print("🚀 [Step 1] 빌드 설정 파일(Spec) 자동 생성 중...")
    try:
        with open(spec_filename, "w", encoding="utf-8") as f:
            f.write(SPEC_CONTENT)
        print(f"✅ '{spec_filename}' 생성 완료.")
    except Exception as e:
        print(f"❌ Spec 파일 생성 실패: {e}")
        return

    print("\n📦 [Step 2] PyInstaller 패키징 진행 중... (시간이 조금 걸립니다)")
    try:
        subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", spec_filename], 
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ 빌드 중 오류가 발생했습니다: {e}")
        return

    print("\n🎉 모든 빌드 및 보안 작업이 완벽하게 완료되었습니다!")
    print("👉 'dist/GODO_SYSTEM/_internal' 폴더에 OrbbecSDK 환경이 완벽히 복제되었습니다.")

if __name__ == "__main__":
    main()