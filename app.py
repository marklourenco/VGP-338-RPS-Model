import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import random
from collections import Counter, defaultdict

# Load Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("rps_model_for_app.pth", map_location=DEVICE)
CLASS_NAMES = checkpoint["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Game Logic
player_history = []
sequence_counter = defaultdict(Counter)
wins = losses = draws = 0

def update_sequence_counter(history):
    "Updates the sequence counter for patterns"
    if len(history) >= 2:
        prev_move = history[-2]
        curr_move = history[-1]
        sequence_counter[prev_move][curr_move] += 1

def predict_next_move(history):
    "Predicts player's next move using weighted sequences"
    if len(history) < 2:
        # Not enough history: fallback to most frequent move
        if len(history) == 0:
            return random.choice(CLASS_NAMES)
        return Counter(history).most_common(1)[0][0]

    last_move = history[-1]
    next_moves = sequence_counter[last_move]

    if next_moves:
        # Predict the most common move after the last move
        predicted_move = next_moves.most_common(1)[0][0]
    else:
        # Fallback if no sequence data
        predicted_move = Counter(history).most_common(1)[0][0]

    return predicted_move

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

# Tkinter App
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

        self.next_round_button = tk.Button(root, text="Next Round", command=self.next_round, state="disabled")
        self.next_round_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack(pady=5)

        self.cap = None
        self.running = False
        self.frame = None
        self.player_move = None

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.running = True
            self.next_round_button.config(state="normal")
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        self.next_round_button.config(state="disabled")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(self.frame)
            img_display = ImageTk.PhotoImage(image=img_pil.resize((640, 480)))
            self.video_label.imgtk = img_display
            self.video_label.config(image=img_display)

        self.root.after(30, self.update_frame)

    def next_round(self):
        global player_history

        if self.frame is None:
            return

        # Transform the current frame for the model
        img_pil = Image.fromarray(self.frame)
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = output.argmax(1).item()
            self.player_move = CLASS_NAMES[pred_idx]

        player_history.append(self.player_move)
        update_sequence_counter(player_history)

        ai_prediction = predict_next_move(player_history)
        ai_move = counter_move(ai_prediction)
        decide_winner(self.player_move, ai_move)

        self.info_label.config(
            text=f"Player: {self.player_move.capitalize()} | AI: {ai_move.capitalize()}"
        )

        self.score_label.config(
            text=f"Wins: {wins} | Losses: {losses} | Draws: {draws}"
        )

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = RPSApp(root)
    root.mainloop()
