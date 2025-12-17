import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import random
from collections import Counter

# -------------------------------
# Load Model
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("rps_model_for_app.pth", map_location=DEVICE)
CLASS_NAMES = checkpoint["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Game Logic
# -------------------------------
player_history = []
wins = losses = draws = 0

def predict_next_move(history):
    if len(history) < 3:
        return random.choice(CLASS_NAMES)
    return Counter(history).most_common(1)[0][0]

def counter_move(move):
    return {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock"
    }[move]

def decide_winner(player, ai):
    global wins, losses, draws

    if player == ai:
        draws += 1
    elif (
        (player == "rock" and ai == "scissors") or
        (player == "paper" and ai == "rock") or
        (player == "scissors" and ai == "paper")
    ):
        wins += 1
    else:
        losses += 1

# -------------------------------
# Tkinter App
# -------------------------------
class RPSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rock Paper Scissors AI")
        self.root.geometry("900x600")

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.info_label = tk.Label(root, text="", font=("Arial", 14))
        self.info_label.pack(pady=10)

        self.score_label = tk.Label(root, text="Wins: 0 | Losses: 0 | Draws: 0", font=("Arial", 14))
        self.score_label.pack()

        self.start_button = tk.Button(root, text="Start Camera", command=self.start_camera)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack(pady=5)

        self.cap = None
        self.running = False

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")

    def update_frame(self):
        global wins, losses, draws

        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = output.argmax(1).item()
            player_move = CLASS_NAMES[pred_idx]

        player_history.append(player_move)

        ai_prediction = predict_next_move(player_history)
        ai_move = counter_move(ai_prediction)

        decide_winner(player_move, ai_move)

        self.info_label.config(
            text=f"Player: {player_move.capitalize()} | AI: {ai_move.capitalize()}"
        )

        self.score_label.config(
            text=f"Wins: {wins} | Losses: {losses} | Draws: {draws}"
        )

        img_display = ImageTk.PhotoImage(image=img_pil.resize((640, 480)))
        self.video_label.imgtk = img_display
        self.video_label.config(image=img_display)

        self.root.after(30, self.update_frame)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = RPSApp(root)
    root.mainloop()
