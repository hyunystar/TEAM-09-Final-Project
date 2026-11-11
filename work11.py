import os
import time
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

SAVE_DIR = "./streetview_dataset"
INTERVAL = 3        # 캡처 간격(초)
START_DELAY = 20    # 시작 전 준비 시간(초): 이 안에 거리뷰 화면까지 이동
MAX_SHOTS = 300     # 최대 장수 (안전장치)

os.makedirs(SAVE_DIR, exist_ok=True)

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--force-device-scale-factor=1")
options.add_argument("--ignore-gpu-blocklist")
options.add_argument("--enable-webgl")

driver = webdriver.Chrome(options=options)
driver.get("https://map.naver.com/p")

print(f"{START_DELAY}초 안에 '둔산동' 검색 → 거리뷰 버튼 → 파란 길 선택까지 해주세요.")
for remaining in range(START_DELAY, 0, -1):
    print(f"  시작까지 {remaining}초", end="\r")
    time.sleep(1)
print("\n캡처 시작! 종료하려면 브라우저 창을 닫거나 터미널에서 Ctrl+C")

count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".png")])
taken = 0

try:
    while taken < MAX_SHOTS:
        count += 1
        taken += 1
        path = os.path.join(SAVE_DIR, f"streetview_{count:04d}.png")
        driver.save_screenshot(path)
        print(f"저장({taken}/{MAX_SHOTS}): {path}")
        time.sleep(INTERVAL)
except KeyboardInterrupt:
    print("\n사용자가 중단했습니다.")
except WebDriverException:
    print("\n브라우저가 닫혀 종료합니다.")
finally:
    try:
        driver.quit()
    except Exception:
        pass
    print(f"총 {taken}장 저장 완료")