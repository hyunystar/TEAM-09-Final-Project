import csv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# --- 설정 ---
MODEL_PATH = "best.pt"
IMAGES_DIR = Path("streetview_dataset")
ANSWERS_CSV = Path("answers.csv")
CONF_THRESHOLD = 0.25
DISPLAY_SIZE = (800, 500)


def load_rounds(csv_path: Path, img_dir: Path):
    rounds = []
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = img_dir / row["filename"].strip()
            if img_path.exists():
                rounds.append({
                    "filename": row["filename"].strip(),
                    "path": img_path,
                    "true_count": int(row["true_count"]),
                })
    return rounds


class GinkgoCountGame:
    def __init__(self, root: tk.Tk, model: YOLO, rounds: list[dict]):
        self.root = root
        self.root.title("사람 vs YOLO 사진 속 은행나무 개수 대결")

        self.model = model
        self.rounds = rounds
        self.idx = 0

        self.human_score = 0
        self.yolo_score = 0
        self.current_photo = None
        self.yolo_count = 0
        self.true_count = 0
        self.submitted = False

        self._build_ui()
        self.load_round()

    def _build_ui(self):
        top_frame = tk.Frame(self.root, pady=8)
        top_frame.pack(fill="x")

        self.lbl_round = tk.Label(top_frame, text="", font=("Malgun Gothic", 12, "bold"))
        self.lbl_round.pack(side="left", padx=16)

        self.lbl_score = tk.Label(
            top_frame, text="", font=("Malgun Gothic", 12, "bold"), fg="#1b5e20"
        )
        self.lbl_score.pack(side="right", padx=16)

        self.lbl_image = tk.Label(self.root, bg="#222222")
        self.lbl_image.pack(padx=12, pady=6)

        self.lbl_question = tk.Label(
            self.root,
            text="이 사진에 보이는 은행나무는 총 몇 그루일까요?",
            font=("Malgun Gothic", 12, "bold"),
        )
        self.lbl_question.pack(pady=4)

        self.user_guess = tk.IntVar(value=0)
        choice_frame = tk.Frame(self.root, pady=4)
        choice_frame.pack()

        for n in range(6):
            label = f"{n}개" if n < 5 else "5개+"
            tk.Radiobutton(
                choice_frame,
                text=label,
                variable=self.user_guess,
                value=n,
                font=("Malgun Gothic", 11),
            ).pack(side="left", padx=8)

        btn_frame = tk.Frame(self.root, pady=8)
        btn_frame.pack()

        self.btn_submit = tk.Button(
            btn_frame,
            text="정답 확인",
            font=("Malgun Gothic", 11, "bold"),
            width=12,
            command=self.check_answer,
        )
        self.btn_submit.pack(side="left", padx=8)

        self.btn_next = tk.Button(
            btn_frame,
            text="다음 라운드",
            font=("Malgun Gothic", 11, "bold"),
            width=12,
            state="disabled",
            command=self.next_round,
        )
        self.btn_next.pack(side="left", padx=8)

        self.lbl_feedback = tk.Label(
            self.root,
            text="",
            font=("Malgun Gothic", 11),
            fg="#0d47a1",
            wraplength=780,
        )
        self.lbl_feedback.pack(pady=(2, 12))

    def _update_scoreboard(self):
        total = len(self.rounds)
        self.lbl_round.config(text=f"라운드: {min(self.idx + 1, total)} / {total}")
        self.lbl_score.config(
            text=f"사람 맞힌 개수: {self.human_score}  |  YOLO 맞힌 개수: {self.yolo_score}"
        )

    def load_round(self):
        self.submitted = False
        self.user_guess.set(0)
        self.btn_submit.config(state="normal")
        self.btn_next.config(state="disabled")
        self.lbl_feedback.config(text="사진 속 은행나무 개수를 고른 뒤 '정답 확인'을 누르세요.")
        self._update_scoreboard()

        current = self.rounds[self.idx]
        self.true_count = current["true_count"]

        results = self.model.predict(source=str(current["path"]), conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        self.yolo_count = len(boxes) if boxes is not None else 0

        img_bgr = cv2.imread(str(current["path"]))
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes.xyxy.cpu().numpy(), start=1):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 215, 255), 3)
                cv2.putText(
                    img_bgr,
                    f"#{i}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 215, 255),
                    2,
                )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(pil_img)
        self.lbl_image.config(image=self.current_photo)

    def check_answer(self):
        if self.submitted:
            return
        self.submitted = True
        self.btn_submit.config(state="disabled")

        guess = self.user_guess.get()
        # 실제 개수를 정확히 맞혔을 때 해당 개수만큼 점수 부여 (원하면 근사치로 변경 가능)
        human_gain = self.true_count if guess == self.true_count else 0
        yolo_gain = self.true_count if self.yolo_count == self.true_count else 0

        self.human_score += human_gain
        self.yolo_score += yolo_gain
        self._update_scoreboard()

        self.lbl_feedback.config(
            text=(
                f"실제 은행나무: {self.true_count}그루 | "
                f"사람 선택 {guess}개 (+{human_gain}점) vs "
                f"YOLO 탐지 {self.yolo_count}개 (+{yolo_gain}점)"
            )
        )

        if self.idx + 1 < len(self.rounds):
            self.btn_next.config(state="normal")
        else:
            self.show_final_result()

    def next_round(self):
        self.idx += 1
        self.load_round()

    def show_final_result(self):
        if self.human_score > self.yolo_score:
            winner = "🎉 사람 승리!"
        elif self.yolo_score > self.human_score:
            winner = "🤖 YOLO 승리!"
        else:
            winner = "🤝 무승부!"

        messagebox.showinfo(
            "최종 결과",
            f"{winner}\n\n"
            f"사람 맞힌 개수: {self.human_score}개\n"
            f"YOLO 맞힌 개수: {self.yolo_score}개",
        )
        self.root.destroy()


if __name__ == "__main__":
    rounds = load_rounds(ANSWERS_CSV, IMAGES_DIR)
    if not rounds:
        raise FileNotFoundError("answers.csv 또는 images/ 폴더에 유효한 라운드 데이터가 없습니다.")

    model = YOLO(MODEL_PATH)
    root = tk.Tk()
    app = GinkgoCountGame(root, model, rounds)
    root.mainloop()