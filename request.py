import os
import time
import requests
import pandas as pd

# 1. 설정 및 API 키 입력
EXCEL_FILE_PATH = "2026 서울시 가로수 위치정보.xlsx" #엑셀 파일명
REST_API_KEY = "629e0ee9acd633f94c5cc79234e0bf58"  # 본인의 카카오 REST API 키
OUTPUT_DIR = "team 09 image"  # 이미지가 저장될 폴더

# 저장 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
  # --- [1단계] 엑셀 파일 읽기 및 은행나무 필터링 ---
  print("엑셀 파일을 불러오는 중...")
  try:
    # 엑셀 파일 읽기 (만약 csv라면 pd.read_csv 사용)
    df = pd.read_excel(EXCEL_FILE_PATH, header=2)
  except Exception as e:
    print(f"엑셀 파일 읽기 실패: {e}")
    return

  # 컬럼명은 공공데이터 양식에 따라 다를 수 있습니다 (예: '수종', '나무종류' 등 확인 필요)
  # '은행나무'가 포함된 행만 추출 (정확히 일치하거나 문자열 포함 조건 적용)
  ginkgo_df = df[df["수종"].str.contains("은행나무", na=False)]

  print(f"총 데이터 중 은행나무 발견: {len(ginkgo_df)}그루")

  # --- [2단계 & 3단계] 위·경도 추출 및 위성 사진 다운로드 루프 ---
  for idx, row in ginkgo_df.iterrows():
    # 엑셀의 위도, 경도 컬럼명에 맞게 수정 (예: '위도', '경도' 또는 'LAT', 'LNG')
    lat = row["좌표(위도)"]
    lng = row["좌표(경도)"]

    # 유효한 좌표인지 확인
    if pd.isna(lat) or pd.isna(lng):
      continue

    # 파일명 지정 (인덱스와 좌표 활용)
    filename = os.path.join(OUTPUT_DIR, f"ginkgo_{idx}_{lat}_{lng}.png")

    # 이미 다운로드된 파일이 있다면 중복 요청 방지 (이어하기 기능)
    if os.path.exists(filename):
      print(f"[{idx+1}/{len(ginkgo_df)}] 이미 존재함: {filename}")
      continue

    # 카카오 Static Map API 요청 URL 설정 (위성 지도 모드)
    url = "https://dapi.kakao.com/v2/maps/staticmap"
    headers = {"Authorization": f"KakaoAK {REST_API_KEY}"}
    params = {
        "center": f"{lng},{lat}",  # 카카오 API는 경도(x), 위도(y) 순서입니다!
        "level": 1,  # 지도 확대 레벨 (숫자가 낮을수록 좁은 지역 확대)
        "size": "640x640",
        "maptype": "hybrid",  # 위성 지도 타입
    }

    try:
      response = requests.get(url, headers=headers, params=params)

      if response.status_code == 200:
        with open(filename, "wb") as f:
          f.write(response.content)
        print(f"[{idx+1}/{len(ginkgo_df)}] 다운로드 성공: {filename}")
      else:
        print(
            f"[{idx+1}/{len(ginkgo_df)}] API 에러 ({response.status_code}):"
            f" {response.text}"
        )

    except Exception as e:
      print(f"[{idx+1}/{len(ginkgo_df)}] 요청 중 예외 발생: {e}")

    # 서버 부하 방지 및 차단(Rate Limit)을 막기 위한 0.5초 휴식
    time.sleep(0.5)

  print("모든 위성 사진 수집 작업이 완료되었습니다!")


if __name__ == "__main__":
  main()