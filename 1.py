pip install playwright pandas pillow requests
playwright install chromium


import os
import re
import csv
import asyncio
import requests
import pandas as pd

from io import BytesIO
from PIL import Image
from playwright.async_api import async_playwright

# 네이버지도 링크
NAVER_MAP_URL = "https://naver.me/539d0FFC"

# 저장 폴더
SAVE_DIR = "naver_photo_dataset"
IMAGE_DIR = os.path.join(SAVE_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

async def collect_images(url):

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False
        )

        page = await browser.new_page()

        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000
        )

        await page.wait_for_timeout(10000)

        # 페이지 내 이미지 주소 수집
        image_urls = await page.locator("img").evaluate_all(
            """imgs => imgs.map(img => ({
                src: img.currentSrc || img.src,
                alt: img.alt
            }))"""
        )

        records = []
        count = 0

        for item in image_urls:

            image_url = item["src"]

            if not image_url.startswith("http"):
                continue

            try:
                response = requests.get(
                    image_url,
                    timeout=15,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

                response.raise_for_status()

                image = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

                if image.width < 150 or image.height < 150:
                    continue

                count += 1

                filename = f"photo_{count:04d}.jpg"
                filepath = os.path.join(
                    IMAGE_DIR,
                    filename
                )

                image.save(filepath, quality=95)

                records.append({
                    "image_id": count,
                    "image_file": filename,
                    "image_url": image_url,
                    "source": url
                })

                print("저장 완료:", filename)

            except Exception:
                continue

        await browser.close()

        # CSV 생성
        df = pd.DataFrame(records)

        df.to_csv(
            os.path.join(SAVE_DIR, "dataset.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        print(f"총 {count}장 저장 완료")

asyncio.run(collect_images(NAVER_MAP_URL))

