"""
🌳 은행나무 구분 대결: 나 vs YOLO
=================================
같은 10문제를 YOLO와 사용자가 동시에 풉니다.
  1) 맞힌 개수가 많은 쪽이 승리
  2) 맞힌 개수가 같으면 더 빨리 끝낸 쪽이 승리

준비물
------
  pip install ultralytics pillow
  - best.pt          : 직접 학습한 YOLO 모델 (이 파일과 같은 폴더)
                       분류(classify) 모델, 탐지(detect) 모델 모두 지원
                       클래스 이름에 'ginkgo' 또는 '은행' 이 들어가면 은행나무로 판단
  - streetview_dataset, streetview_dataset2
                     (네이버 거리뷰 캡처. 이 파일과 같은 폴더에 폴더 또는 .zip 으로 두기)
                     정답 라벨은 아래 중 어느 방식이든 자동 인식합니다.
                       ⓪ labels.csv (path,label) : 시작 화면의 '📝 라벨링' 버튼으로 만들 수 있음
                          (거리뷰 사진만 있고 라벨이 없을 때 사용. 키보드 ←/→ 로 600장도 금방 끝남)
                       ① YOLO 형식: data.yaml(names) + labels/*.txt
                          → 은행나무 클래스 박스가 있으면 '은행나무', 없으면 '다른 나무'
                       ② 폴더명/파일명에 ginkgo·은행 (은행나무) 또는
                          other·non·다른·기타 (다른 나무) 가 들어간 경우
                     test/val 폴더가 있으면 학습에 안 쓴 사진으로 출제합니다.
                     다른 위치에 있으면:  python ginkgo_battle.py 경로1 경로2

  best.pt 나 데이터셋이 없으면 데모 모드로 동작합니다.
  (모의 YOLO + 그림으로 만든 데모 잎 이미지)

실행
----
  python ginkgo_battle.py
조작: 왼쪽 버튼/←/1 = 은행나무,  오른쪽 버튼/→/2 = 다른 나무
"""
import math
import os
import queue
import random
import re
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFilter

try:  # GUI 가 없는 환경(테스트 등)에서도 로직은 import 가능하도록
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk
    from PIL import ImageTk
except Exception:  # pragma: no cover
    tk = None

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAMES = ("streetview_dataset", "streetview_dataset2")   # 폴더 또는 .zip
PREFER_UNSEEN_SPLITS = True        # test/val 폴더가 있으면 그 사진으로만 출제
UNSEEN_SPLIT_NAMES = ("test", "val", "valid", "validation")
NEG_KEYWORDS = ("non", "not", "other", "no_", "no-", "etc", "negative", "neg_",
                "다른", "기타", "아님", "비은행")
CLASS_FILE_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "classes.txt", "obj.names", "names.txt")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
NUM_QUESTIONS = 10
GINKGO_KEYWORDS = ("ginkgo", "gingko", "은행")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMG_BOX = (940, 430)                              # 화면에 표시할 최대 이미지 크기 (거리뷰는 가로로 넓음)
MASK_UI = True                                    # 거리뷰 UI(지도·주소창·버튼) 가리기
# 1920x889 캡처 기준 UI 영역 (왼쪽, 위, 오른쪽, 아래: 비율)
UI_RECTS = ((0.0, 0.69, 0.18, 1.0),               # 왼쪽 아래 주소창 + 미니맵
            (0.85, 0.0, 1.0, 0.075),              # 오른쪽 위 '장소·방면정보' 토글, 닫기
            (0.96, 0.33, 1.0, 0.68))              # 오른쪽 나침반/버튼들
LABEL_CSV = "labels.csv"                          # 각 데이터셋 폴더 안에 저장되는 라벨 파일

# 모의 YOLO(데모 모드) 설정
SIM_ACCURACY = 0.85
SIM_DELAY = (0.4, 1.4)                            # 문제당 풀이 시간(초)

GINKGO, OTHER = "ginkgo", "other"
LABEL_KO = {GINKGO: "은행나무", OTHER: "다른 나무"}


# ----------------------------------------------------------------------------
# 게임 로직 (GUI 와 무관)
# ----------------------------------------------------------------------------
@dataclass
class Question:
    image: Image.Image
    label: str                      # GINKGO / OTHER
    path: Optional[str] = None      # 실제 파일이면 경로
    full: Optional[Image.Image] = None   # UI 를 가린 원본 해상도 이미지 (YOLO 입력용)


def judge(user_correct, yolo_correct, user_time, yolo_time):
    """(승자, 사유) 반환. 승자: 'user' / 'yolo' / 'draw'"""
    if user_correct != yolo_correct:
        return ("user" if user_correct > yolo_correct else "yolo"), "정답 수"
    if abs(user_time - yolo_time) < 1e-9:
        return "draw", "정답 수·속도 모두 동일"
    return ("user" if user_time < yolo_time else "yolo"), "속도"


def find_dataset_dirs(extra=()):
    """데이터셋 폴더를 찾는다. zip 이면 옆에 풀어서 사용."""
    cands = [os.path.abspath(p) for p in extra]
    for base in dict.fromkeys((BASE_DIR, os.getcwd(), os.path.dirname(BASE_DIR))):
        for nm in DATASET_NAMES:
            d = os.path.join(base, nm)
            z = d + ".zip"
            if not os.path.isdir(d) and os.path.isfile(z):
                try:
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(d)
                except Exception:
                    pass
            cands.append(d)
    cands.append(os.path.join(BASE_DIR, "dataset"))
    roots, seen = [], set()
    for c in cands:
        c = os.path.abspath(c)
        if os.path.isdir(c) and c not in seen:
            seen.add(c)
            roots.append(c)
    return roots


def _is_ginkgo_name(name):
    name = str(name).lower()
    if any(k in name for k in NEG_KEYWORDS):
        return False
    return any(k in name for k in GINKGO_KEYWORDS)


def read_class_names(root):
    """data.yaml / classes.txt 등에서 클래스 이름 리스트를 읽는다. 없으면 None."""
    for dirpath, dirnames, files in os.walk(root):
        if dirpath[len(root):].count(os.sep) > 2:
            dirnames[:] = []
            continue
        for fn in files:
            if fn.lower() not in CLASS_FILE_NAMES:
                continue
            p = os.path.join(dirpath, fn)
            try:
                text = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if fn.lower().endswith((".yaml", ".yml")):
                names = None
                try:
                    import yaml
                    names = (yaml.safe_load(text) or {}).get("names")
                except Exception:
                    m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
                    if m:
                        names = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
                    else:
                        m = re.search(r"names\s*:\s*\n((?:\s+.+\n?)+)", text)
                        if m:
                            items = re.findall(r"^\s*(?:(\d+)\s*:|-)\s*(.+?)\s*$", m.group(1), re.M)
                            names = {int(i) if i else k: v.strip("'\"") for k, (i, v) in enumerate(items)}
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names)]
                if names:
                    return [str(n) for n in names]
            else:
                names = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if names:
                    return names
    return None


def _label_file(img_path):
    sep = os.sep
    sp = img_path.replace(sep + "images" + sep, sep + "labels" + sep)
    for c in (os.path.splitext(sp)[0] + ".txt", os.path.splitext(img_path)[0] + ".txt"):
        if os.path.isfile(c):
            return c
    return None


def resolve_label(img_path, root, ginkgo_ids):
    """사진 한 장의 정답을 결정. 알 수 없으면 None."""
    # ① YOLO 라벨 (클래스 이름에서 은행나무 id 를 알 때)
    if ginkgo_ids:
        lf = _label_file(img_path)
        if lf:
            try:
                ids = set()
                for ln in open(lf, encoding="utf-8", errors="ignore"):
                    tok = ln.split()
                    if tok:
                        ids.add(int(float(tok[0])))
                return GINKGO if ids & ginkgo_ids else OTHER
            except ValueError:
                pass
        elif (os.sep + "images" + os.sep) in img_path:   # YOLO 구조에서 라벨 없음 = 배경 = 다른 나무
            return OTHER
    # ② 폴더명/파일명 키워드
    rel = os.path.relpath(img_path, root).lower().replace("\\", "/")
    if any(k in rel for k in NEG_KEYWORDS):
        return OTHER
    if any(k in rel for k in GINKGO_KEYWORDS):
        return GINKGO
    return None


_G_TOK = {"ginkgo", "g", "1", "은행", "은행나무"}
_O_TOK = {"other", "o", "0", "다른", "다른나무", "다른 나무", "non", "non_ginkgo"}


def _rel(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


def load_csv(root):
    """root/labels.csv (path,label) 을 읽는다. {상대경로: 라벨}"""
    p = os.path.join(root, LABEL_CSV)
    m = {}
    if not os.path.isfile(p):
        return m
    import csv
    with open(p, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            tok = row[1].strip().lower()
            if tok in _G_TOK:
                m[row[0].strip()] = GINKGO
            elif tok in _O_TOK:
                m[row[0].strip()] = OTHER
    return m


class DatasetIndex:
    def __init__(self):
        self.items = []          # (경로, 라벨, 미학습 분할 여부)
        self.roots = []
        self.skipped = 0
        self.unlabeled = []      # [(경로, 루트)] 라벨링이 필요한 사진
        self.csv = {}            # {루트: {상대경로: 라벨}}

    # --- 라벨링 도구용 ---
    def _save_csv(self, root):
        import csv
        with open(os.path.join(root, LABEL_CSV), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            for k, v in sorted(self.csv.get(root, {}).items()):
                w.writerow([k, v])

    def set_label(self, path, root, label):
        self.csv.setdefault(root, {})[_rel(path, root)] = label
        self.items.append((path, label, False))
        self._save_csv(root)

    def remove_label(self, path, root):
        self.csv.get(root, {}).pop(_rel(path, root), None)
        self.items = [i for i in self.items if i[0] != path]
        self._save_csv(root)

    def pool(self, prefer_unseen=PREFER_UNSEEN_SPLITS):
        need = NUM_QUESTIONS // 2
        if prefer_unseen:
            u = [i for i in self.items if i[2]]
            if (sum(i[1] == GINKGO for i in u) >= need and sum(i[1] == OTHER for i in u) >= need):
                return u, True
        return self.items, False

    def lists(self):
        items, unseen = self.pool()
        return ([p for p, l, _ in items if l == GINKGO], [p for p, l, _ in items if l == OTHER], unseen)

    @property
    def summary(self):
        if not self.roots:
            return "데이터셋 폴더를 찾지 못함 → 데모 이미지 사용"
        g, o, unseen = self.lists()
        names = ", ".join(os.path.basename(r) for r in self.roots)
        s = f"데이터셋 [{names}]: 은행나무 {len(g)}장 / 다른 나무 {len(o)}장"
        if unseen:
            s += " (test/val 사진만 사용)"
        if self.unlabeled:
            s += f" · 미라벨 {len(self.unlabeled)}장 (아래 '라벨링' 버튼으로 정답 입력)"
        return s


def build_index(roots):
    idx = DatasetIndex()
    idx.roots = list(roots)
    for root in roots:
        names = read_class_names(root)
        gids = {i for i, n in enumerate(names or []) if _is_ginkgo_name(n)}
        idx.csv[root] = load_csv(root)
        for dirpath, _, files in os.walk(root):
            parts = {p.lower() for p in os.path.relpath(dirpath, root).split(os.sep)}
            unseen = bool(parts & set(UNSEEN_SPLIT_NAMES))
            for fn in sorted(files):
                if not fn.lower().endswith(IMG_EXTS):
                    continue
                p = os.path.join(dirpath, fn)
                lab = idx.csv[root].get(_rel(p, root)) or resolve_label(p, root, gids)
                if lab is None:
                    idx.skipped += 1
                    idx.unlabeled.append((p, root))
                else:
                    idx.items.append((p, lab, unseen))
    return idx


def mask_ui(img):
    """네이버 거리뷰 캡처의 UI 영역을 회색으로 가린다 (사람·YOLO 모두 같은 이미지를 본다)."""
    img = img.convert("RGB")
    W, H = img.size
    if not MASK_UI or abs(W / H - 1920 / 889) > 0.15:
        return img
    img = img.copy()
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in UI_RECTS:
        d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], fill=(110, 110, 110))
    return img


def _fit(img, box=IMG_BOX):
    img = img.convert("RGB")
    img.thumbnail(box, Image.LANCZOS)
    return img


def _poly_arc(cx, cy, r, a0, a1, n=40, notch=0):
    pts = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        rr = r
        if notch and abs(math.degrees(a)) < 7:
            rr = r - notch
        pts.append((cx + rr * math.sin(a), cy - rr * math.cos(a)))
    return pts


def make_demo_image(label, rng):
    """데모용 잎 그림. 은행잎(부채꼴+노랑) vs 다른 잎(단풍/참나무/타원)."""
    W, H = IMG_BOX
    bg = Image.new("RGB", (W, H), (rng.randint(60, 110), rng.randint(70, 100), rng.randint(40, 70)))
    bg = bg.filter(ImageFilter.GaussianBlur(2))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = W // 2, H // 2 + 20

    if label == GINKGO:
        col = (rng.randint(215, 245), rng.randint(190, 215), rng.randint(30, 70), 255)
        base = (cx, cy + 130)
        arc = _poly_arc(cx, cy + 90, 170, -62, 62, notch=35)
        d.polygon([base] + arc, fill=col, outline=(150, 120, 20, 255))
        for a in range(-55, 60, 12):
            ar = math.radians(a)
            d.line([base, (cx + 150 * math.sin(ar), cy + 90 - 150 * math.cos(ar))],
                   fill=(190, 160, 30, 255), width=1)
        d.line([base, (cx, cy + 190)], fill=(110, 90, 30, 255), width=5)
    else:
        kind = rng.choice(["maple", "oak", "oval"])
        if kind == "maple":
            col = rng.choice([(200, 50, 30), (225, 120, 30), (70, 140, 50)]) + (255,)
            pts = []
            for i in range(10):
                a = math.radians(i * 36 - 90)
                r = 150 if i % 2 == 0 else 65
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            d.polygon(pts, fill=col, outline=(80, 30, 10, 255))
        elif kind == "oak":
            col = (rng.randint(60, 110), rng.randint(120, 160), rng.randint(40, 70), 255)
            for i in range(-3, 4):
                w = 70 - abs(i) * 8
                d.ellipse([cx - w, cy + i * 38 - 28, cx + w, cy + i * 38 + 28], fill=col,
                          outline=(30, 80, 20, 255))
        else:
            col = (rng.randint(40, 90), rng.randint(130, 180), rng.randint(50, 90), 255)
            d.polygon([(cx, cy - 160), (cx + 85, cy), (cx, cy + 160), (cx - 85, cy)], fill=col,
                      outline=(20, 70, 20, 255))
            d.line([(cx, cy - 160), (cx, cy + 190)], fill=(30, 90, 30, 255), width=3)
    layer = layer.rotate(rng.randint(-30, 30), resample=Image.BICUBIC, center=(cx, cy))
    bg.paste(layer, (0, 0), layer)
    return bg


def _pick(paths, k, label, rng):
    """열 수 있는 사진 k장을 무작위로 뽑는다 (깨진 파일은 건너뜀)."""
    out = []
    for p in rng.sample(paths, len(paths)):
        if len(out) >= k:
            break
        try:
            with Image.open(p) as im:
                full = mask_ui(im)
            out.append(Question(_fit(full), label, p, full))
        except Exception:
            continue
    return out


def load_questions(index=None, n=NUM_QUESTIONS, rng=random):
    """(질문 리스트, 'real'|'demo') 반환."""
    g, o = (index.lists()[:2] if index else ([], []))
    n_g = min(n // 2, len(g))
    n_o = min(n - n_g, len(o))
    n_g = min(n - n_o, len(g))
    qs = []
    if n_g + n_o == n:
        qs = _pick(g, n_g, GINKGO, rng) + _pick(o, n_o, OTHER, rng)
    mode = "real"
    if len(qs) != n:
        qs, mode = [], "demo"
    if mode == "demo":
        k = n // 2
        for i in range(n):
            lab = GINKGO if i < k else OTHER
            qs.append(Question(make_demo_image(lab, rng), lab))
        mode = "demo"
    rng.shuffle(qs)
    return qs, mode


class YoloPlayer:
    """YOLO 모델 래퍼. 모델이 없으면 모의 플레이어로 동작."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.simulated = True
        self.note = ""

    def load(self):
        if not os.path.exists(self.model_path):
            self.note = "best.pt 없음 → 모의 YOLO(데모)"
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            names = [str(v).lower() for v in self.model.names.values()]
            if not any(k in nm for nm in names for k in GINKGO_KEYWORDS):
                self.note = f"주의: 클래스 이름({names})에 ginkgo/은행 이 없어 모두 '다른 나무'로 판단될 수 있음"
            else:
                self.note = f"YOLO 모델 로드 완료 ({os.path.basename(self.model_path)})"
            self.model.predict(Image.new("RGB", (224, 224)), verbose=False)  # 워밍업
            self.simulated = False
        except Exception as e:  # ultralytics 미설치/모델 오류
            self.model = None
            self.simulated = True
            self.note = f"YOLO 로드 실패({type(e).__name__}) → 모의 YOLO(데모)"

    def _is_ginkgo_name(self, name):
        return _is_ginkgo_name(name)     # 'non_ginkgo' 같은 이름은 다른 나무로 처리

    def predict(self, q, rng=random):
        if self.simulated:
            time.sleep(rng.uniform(*SIM_DELAY))
            return q.label if rng.random() < SIM_ACCURACY else (OTHER if q.label == GINKGO else GINKGO)
        src = q.full if q.full is not None else (q.path or q.image)
        r = self.model.predict(src, verbose=False)[0]
        if getattr(r, "probs", None) is not None:           # 분류 모델
            name = r.names[int(r.probs.top1)]
        elif r.boxes is not None and len(r.boxes) > 0:      # 탐지 모델: 가장 확신 높은 박스
            i = int(r.boxes.conf.argmax())
            name = r.names[int(r.boxes.cls[i])]
        else:
            return OTHER
        return GINKGO if self._is_ginkgo_name(name) else OTHER


def yolo_worker(player, questions, t0, out_q):
    """별도 스레드에서 10문제를 순서대로 풀며 진행 상황을 큐로 보고."""
    for i, q in enumerate(questions):
        ans = player.predict(q)
        out_q.put(("progress", i, ans, time.perf_counter() - t0))
    out_q.put(("done",))


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
BG, PANEL, FG = "#1f2a24", "#2b3a32", "#f1f5ef"
GREEN, YELLOW, RED, GRAY = "#5cb85c", "#f0c040", "#e05555", "#8a9a8f"


class App(tk.Tk if tk else object):
    def __init__(self):
        super().__init__()
        self.title("🌳 은행나무 대결: 나 vs YOLO")
        self.geometry("1000x880")
        self.configure(bg=BG)
        self.resizable(False, False)

        fams = set(tkfont.families())
        fam = next((f for f in ("Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic",
                                "Noto Sans CJK KR", "Noto Sans KR") if f in fams), "TkDefaultFont")
        self.F = lambda size, bold=False: (fam, size, "bold" if bold else "normal")

        self.yolo = YoloPlayer()
        self.yolo_ready = False
        self.index = None
        threading.Thread(target=self._load_model, daemon=True).start()

        self.state = "start"
        self.frame = None
        self.bind_all("<Key>", self._on_key)
        self.show_start()
        self.after(50, self._poll)

    # ---- 공통 ------------------------------------------------------------
    def _load_model(self):
        self.index = build_index(find_dataset_dirs(sys.argv[1:]))
        self.yolo.load()
        self.yolo_ready = True

    def _new_frame(self):
        if self.frame is not None:
            self.frame.destroy()
        self.frame = tk.Frame(self, bg=BG)
        self.frame.pack(fill="both", expand=True, padx=20, pady=16)
        return self.frame

    def _btn(self, parent, text, cmd, color, **kw):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white", activebackground=color,
                         relief="flat", bd=0, cursor="hand2", font=self.F(16, True), **kw)

    # ---- 시작 화면 -------------------------------------------------------
    def show_start(self):
        self.state = "start"
        f = self._new_frame()
        tk.Label(f, text="🌳 vs 🤖", font=self.F(44), bg=BG, fg=FG).pack(pady=(60, 6))
        tk.Label(f, text="은행나무 구분 대결", font=self.F(30, True), bg=BG, fg=YELLOW).pack()
        rules = ("YOLO와 내가 같은 10문제를 동시에 풉니다.\n"
                 "은행나무인지, 다른 나무인지 맞혀보세요!\n\n"
                 "🏆 맞힌 개수가 많은 쪽이 승리\n"
                 "⚡ 개수가 같으면 더 빨리 끝낸 쪽이 승리\n\n"
                 "조작: ←/1 = 은행나무    →/2 = 다른 나무")
        tk.Label(f, text=rules, font=self.F(14), bg=BG, fg=FG, justify="center").pack(pady=26)
        self.start_status = tk.Label(f, text="YOLO·데이터셋 준비 중...", font=self.F(12), bg=BG, fg=GRAY,
                                     justify="center", wraplength=700)
        self.start_status.pack(pady=(0, 10))
        self.start_btn = self._btn(f, "게임 시작", self.start_game, GREEN, width=16, pady=10, state="disabled")
        self.start_btn.pack()
        self.label_btn = self._btn(f, "📝 라벨링", self.show_labeler, "#666f69", width=24, pady=6, state="disabled")
        self.label_btn.config(font=self.F(12, True))
        self.label_btn.pack(pady=(12, 0))

    # ---- 라벨링 도구 -----------------------------------------------------
    def show_labeler(self):
        """미라벨 사진을 하나씩 보며 정답 입력. 결과는 각 데이터셋 폴더의 labels.csv 에 즉시 저장."""
        self.state = "label"
        self.todo = list(self.index.unlabeled)
        self.lab_pos = 0
        self.lab_hist = []          # [(위치, 라벨했는가)]
        f = self._new_frame()
        self.lab_head = tk.Label(f, text="", font=self.F(16, True), bg=BG, fg=FG)
        self.lab_head.pack(anchor="w")
        tk.Label(f, text="기준: 사진에 은행나무(노란 부채꼴 잎/은행나무 수형)가 뚜렷이 보이면 '은행나무', 보이지 않으면 '다른 나무'",
                 font=self.F(11), bg=BG, fg=GRAY).pack(anchor="w")
        self.lab_img = tk.Label(f, bg=PANEL)
        self.lab_img.pack(pady=12, fill="x")
        row = tk.Frame(f, bg=BG)
        row.pack(fill="x")
        for text, color, val in (("🍂 은행나무 (←/G)", "#d9a521", GINKGO), ("🌿 다른 나무 (→/O)", "#3f8f5f", OTHER),
                                 ("건너뛰기 (S)", "#666f69", None)):
            b = self._btn(row, text, lambda v=val: self._lab_set(v), color, pady=12)
            b.pack(side="left", expand=True, fill="x", padx=4)
        row2 = tk.Frame(f, bg=BG)
        row2.pack(pady=14)
        self._btn(row2, "↩ 되돌리기 (Backspace)", self._lab_undo, "#666f69", pady=6, width=22).pack(side="left", padx=6)
        self._btn(row2, "저장하고 돌아가기", self._lab_done, GREEN, pady=6, width=22).pack(side="left", padx=6)
        self._lab_show()

    def _lab_show(self):
        g = sum(1 for i in self.index.items if i[1] == GINKGO)
        o = sum(1 for i in self.index.items if i[1] == OTHER)
        n = len(self.todo)
        if self.lab_pos >= n:
            self.lab_img.config(image="", text="🎉 모두 라벨링했습니다!\n'저장하고 돌아가기'를 누르세요",
                                font=self.F(26, True), fg=YELLOW, height=10)
            self.photo = None
            self.lab_head.config(text=f"완료  ·  은행나무 {g}장 / 다른 나무 {o}장")
            return
        path, _ = self.todo[self.lab_pos]
        try:
            with Image.open(path) as im:
                self.photo = ImageTk.PhotoImage(_fit(mask_ui(im)))
            self.lab_img.config(image=self.photo, text="", height=0)
        except Exception:
            self.lab_img.config(image="", text=f"열 수 없는 파일\n{os.path.basename(path)}", fg=RED)
            self.photo = None
        self.lab_head.config(text=f"라벨링 {self.lab_pos + 1} / {n}  ·  은행나무 {g}장 / 다른 나무 {o}장")

    def _lab_set(self, label):
        if self.lab_pos >= len(self.todo):
            return
        path, root = self.todo[self.lab_pos]
        if label is not None:
            self.index.set_label(path, root, label)
        self.lab_hist.append((self.lab_pos, label is not None))
        self.lab_pos += 1
        self._lab_show()

    def _lab_undo(self):
        if not self.lab_hist:
            return
        pos, labeled = self.lab_hist.pop()
        if labeled:
            path, root = self.todo[pos]
            self.index.remove_label(path, root)
        self.lab_pos = pos
        self._lab_show()

    def _lab_done(self):
        self.index = build_index(self.index.roots)      # 저장된 labels.csv 로 다시 읽기
        self.show_start()

    # ---- 게임 시작 -------------------------------------------------------
    def start_game(self):
        self.questions, self.data_mode = load_questions(self.index)
        self.N = len(self.questions)
        self.idx = 0
        self.user_answers, self.user_times = [], []
        self.yolo_answers, self.yolo_times = [], []
        self.yolo_done = False
        self.user_time = None
        self.yolo_q = queue.Queue()
        self.photo = None
        self._build_quiz()
        self.state = "countdown"
        self._countdown(3)

    def _build_quiz(self):
        f = self._new_frame()
        top = tk.Frame(f, bg=BG)
        top.pack(fill="x")
        self.lbl_q = tk.Label(top, text="", font=self.F(18, True), bg=BG, fg=FG)
        self.lbl_q.pack(side="left")
        self.lbl_timer = tk.Label(top, text="⏱ 0.00초", font=self.F(18, True), bg=BG, fg=YELLOW)
        self.lbl_timer.pack(side="right")

        self.img_area = tk.Label(f, bg=PANEL, fg=YELLOW, font=self.F(80, True),
                                 width=IMG_BOX[0] // 8, height=IMG_BOX[1] // 16 + 2)
        self.img_area.pack(pady=14, fill="x")

        row = tk.Frame(f, bg=BG)
        row.pack(fill="x")
        self.btn_g = self._btn(row, "🍂 은행나무  (←)", lambda: self.answer(GINKGO), "#d9a521", pady=14)
        self.btn_o = self._btn(row, "🌿 다른 나무  (→)", lambda: self.answer(OTHER), "#3f8f5f", pady=14)
        self.btn_g.pack(side="left", expand=True, fill="x", padx=(0, 8))
        self.btn_o.pack(side="left", expand=True, fill="x", padx=(8, 0))

        bars = tk.Frame(f, bg=PANEL)
        bars.pack(fill="x", pady=(18, 0), ipady=8)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("me.Horizontal.TProgressbar", troughcolor=BG, background=GREEN)
        style.configure("ai.Horizontal.TProgressbar", troughcolor=BG, background="#5aa9e6")
        self.lbl_me = tk.Label(bars, text="🙂 나: 0/%d" % self.N, font=self.F(13), bg=PANEL, fg=FG, anchor="w")
        self.lbl_me.pack(fill="x", padx=14, pady=(6, 2))
        self.bar_me = ttk.Progressbar(bars, maximum=self.N, style="me.Horizontal.TProgressbar")
        self.bar_me.pack(fill="x", padx=14)
        kind = "모의 YOLO" if self.yolo.simulated else "YOLO"
        self.lbl_ai = tk.Label(bars, text=f"🤖 {kind}: 0/{self.N}", font=self.F(13), bg=PANEL, fg=FG, anchor="w")
        self.lbl_ai.pack(fill="x", padx=14, pady=(10, 2))
        self.bar_ai = ttk.Progressbar(bars, maximum=self.N, style="ai.Horizontal.TProgressbar")
        self.bar_ai.pack(fill="x", padx=14)
        self.lbl_note = tk.Label(f, text="", font=self.F(11), bg=BG, fg=GRAY)
        self.lbl_note.pack(pady=(10, 0))

    def _countdown(self, n):
        if n > 0:
            self.img_area.config(text=str(n), image="")
            self.lbl_q.config(text="준비...")
            self.after(800, lambda: self._countdown(n - 1))
            return
        self.img_area.config(text="")
        self.t0 = time.perf_counter()
        self.state = "playing"
        threading.Thread(target=yolo_worker, args=(self.yolo, self.questions, self.t0, self.yolo_q),
                         daemon=True).start()   # YOLO 와 사용자가 동시에 출발!
        self.show_question()

    def show_question(self):
        q = self.questions[self.idx]
        self.photo = ImageTk.PhotoImage(q.image)
        self.img_area.config(image=self.photo, text="")
        self.lbl_q.config(text=f"문제 {self.idx + 1} / {self.N}")

    # ---- 입력 ------------------------------------------------------------
    def _on_key(self, e):
        if self.state == "label":
            k = e.keysym.lower()
            if k in ("left", "g", "1"):
                self._lab_set(GINKGO)
            elif k in ("right", "o", "2"):
                self._lab_set(OTHER)
            elif k in ("down", "s", "space"):
                self._lab_set(None)
            elif k == "backspace":
                self._lab_undo()
            return
        if self.state != "playing":
            return
        if e.keysym in ("Left", "1"):
            self.answer(GINKGO)
        elif e.keysym in ("Right", "2"):
            self.answer(OTHER)

    def answer(self, choice):
        if self.state != "playing":
            return
        self.user_answers.append(choice)
        self.user_times.append(time.perf_counter() - self.t0)
        self.idx += 1
        self.bar_me["value"] = self.idx
        self.lbl_me.config(text=f"🙂 나: {self.idx}/{self.N}")
        if self.idx >= self.N:
            self.user_time = self.user_times[-1]
            self.state = "waiting"
            self.lbl_timer.config(text=f"⏱ {self.user_time:.2f}초 (완료)")
            self.img_area.config(image="", text="🤖 YOLO 마무리 중...", font=self.F(24, True))
            self.photo = None
        else:
            self.show_question()

    # ---- 주기적 갱신 -------------------------------------------------------
    def _poll(self):
        # 시작 화면: 모델 준비 상태
        if self.state == "start" and getattr(self, "start_btn", None) is not None:
            try:
                if self.yolo_ready:
                    ok = not self.yolo.simulated and self.index.items
                    self.start_status.config(text=self.yolo.note + "\n" + self.index.summary,
                                             fg=GREEN if ok else YELLOW)
                    self.start_btn.config(state="normal")
                    n = len(self.index.unlabeled)
                    self.label_btn.config(state="normal" if n else "disabled",
                                          text=f"📝 라벨링 (미라벨 {n}장)" if n else "📝 라벨링 (모두 완료)")
            except tk.TclError:
                pass
        # 게임 중: YOLO 진행 수신, 타이머
        if self.state in ("playing", "waiting"):
            try:
                while True:
                    msg = self.yolo_q.get_nowait()
                    if msg[0] == "progress":
                        _, i, ans, t = msg
                        self.yolo_answers.append(ans)
                        self.yolo_times.append(t)
                        self.bar_ai["value"] = len(self.yolo_answers)
                        kind = "모의 YOLO" if self.yolo.simulated else "YOLO"
                        self.lbl_ai.config(text=f"🤖 {kind}: {len(self.yolo_answers)}/{self.N}")
                    else:
                        self.yolo_done = True
            except queue.Empty:
                pass
            if self.state == "playing":
                self.lbl_timer.config(text=f"⏱ {time.perf_counter() - self.t0:.2f}초")
            if self.state == "waiting" and self.yolo_done:
                self.show_result()
        self.after(50, self._poll)

    # ---- 결과 화면 -------------------------------------------------------
    def show_result(self):
        self.state = "result"
        qs = self.questions
        my_ok = [a == q.label for a, q in zip(self.user_answers, qs)]
        ai_ok = [a == q.label for a, q in zip(self.yolo_answers, qs)]
        my_c, ai_c = sum(my_ok), sum(ai_ok)
        ai_time = self.yolo_times[-1]
        winner, why = judge(my_c, ai_c, self.user_time, ai_time)

        f = self._new_frame()
        banner = {"user": ("🎉 당신의 승리!", GREEN),
                  "yolo": ("🤖 YOLO의 승리!", "#5aa9e6"),
                  "draw": ("🤝 무승부", YELLOW)}[winner]
        tk.Label(f, text=banner[0], font=self.F(34, True), bg=BG, fg=banner[1]).pack(pady=(6, 0))
        sub = f"결정 기준: {why}" if winner != "draw" else why
        tk.Label(f, text=sub, font=self.F(13), bg=BG, fg=GRAY).pack()

        cards = tk.Frame(f, bg=BG)
        cards.pack(fill="x", pady=14)
        kind = "모의 YOLO" if self.yolo.simulated else "YOLO"
        for title, c, t, win in (("🙂 나", my_c, self.user_time, winner == "user"),
                                 (f"🤖 {kind}", ai_c, ai_time, winner == "yolo")):
            card = tk.Frame(cards, bg=PANEL, highlightthickness=3,
                            highlightbackground=banner[1] if win else PANEL)
            card.pack(side="left", expand=True, fill="both", padx=6)
            tk.Label(card, text=title, font=self.F(15, True), bg=PANEL, fg=FG).pack(pady=(10, 0))
            tk.Label(card, text=f"{c} / {self.N} 정답", font=self.F(24, True), bg=PANEL, fg=YELLOW).pack()
            tk.Label(card, text=f"⏱ {t:.2f}초", font=self.F(15), bg=PANEL, fg=FG).pack(pady=(0, 10))

        cols = ("no", "ans", "me", "ai")
        style = ttk.Style()
        style.configure("Treeview", font=self.F(12), rowheight=28, background=PANEL,
                        fieldbackground=PANEL, foreground=FG)
        style.configure("Treeview.Heading", font=self.F(12, True))
        tv = ttk.Treeview(f, columns=cols, show="headings", height=self.N)
        for c, h, w in (("no", "#", 40), ("ans", "정답", 130), ("me", "내 답 (누적시간)", 230),
                        ("ai", f"{kind} 답 (누적시간)", 230)):
            tv.heading(c, text=h)
            tv.column(c, width=w, anchor="center")
        tv.tag_configure("ok", foreground="#8fe08f")
        tv.tag_configure("bad", foreground="#ff8f8f")
        for i, q in enumerate(qs):
            mark = lambda ok: "✅" if ok else "❌"
            tv.insert("", "end", values=(
                i + 1, LABEL_KO[q.label],
                f"{mark(my_ok[i])} {LABEL_KO[self.user_answers[i]]} ({self.user_times[i]:.1f}s)",
                f"{mark(ai_ok[i])} {LABEL_KO[self.yolo_answers[i]]} ({self.yolo_times[i]:.1f}s)"))
        tv.pack(fill="x")

        if self.data_mode == "demo":
            tk.Label(f, text="※ 데모 이미지 사용 중 (streetview_dataset 에서 은행나무·다른 나무 사진을 각각 5장 이상 찾지 못함)",
                     font=self.F(10), bg=BG, fg=GRAY).pack(pady=(8, 0))
        row = tk.Frame(f, bg=BG)
        row.pack(pady=14)
        self._btn(row, "다시 하기", self.start_game, GREEN, width=12, pady=8).pack(side="left", padx=8)
        self._btn(row, "종료", self.destroy, "#666f69", width=12, pady=8).pack(side="left", padx=8)


def main():
    if tk is None:
        raise SystemExit("tkinter 를 불러올 수 없습니다. (Linux: sudo apt install python3-tk)")
    App().mainloop()


if __name__ == "__main__":
    main()