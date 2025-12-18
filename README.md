# Rock Paper Scissors AI Desktop App

## Project Overview

This project is a **desktop AI application** that plays Rock Paper Scissors against a human player using **computer vision and machine learning**.
The AI detects hand gestures using a trained deep learning model and adapts to player behavior by learning patterns in their previous moves.

The application runs as a **Windows desktop executable**, requiring no Python installation on the user’s machine.

---

## Features

* Real-time webcam hand gesture recognition
* AI opponent that adapts to player behavior
* Win/Loss/Draw tracking
* Simple, beginner-friendly desktop UI
* Fully packaged `.exe` application

---

## Technologies Used

* **Python 3.10**
* **PyTorch**
* **ResNet-34**
* **OpenCV**
* **Tkinter**
* **PyInstaller**
* **Google Colab** (model training)

---

## Setup Instructions

### 1. Install Python (Development Only)

Download Python 3.10 from:
[https://www.python.org/downloads/]([https://www.python.org/downloads/](https://www.python.org/downloads/release/python-3104/))

---

### 2. Install Required Libraries

```bash
python -m pip install torch torchvision opencv-python pillow numpy
```

---

### 3. Run the Application

```bash
python app.py
```

---

## Running the Executable

1. Open the provided folder
2. Double-click `app.exe`
3. Click **Start Camera**
4. Press **Next Round** to play each round

> Webcam access is required

---

## Model Selection Justification

### Model Used: **ResNet-34**

**Why ResNet-34?**

* Strong performance on image classification tasks
* Uses residual connections to prevent vanishing gradients
* Lightweight enough for real-time webcam inference
* Well-supported and pretrained architecture
---

## Training Process

* Dataset: Rock Paper Scissors image dataset
* Classes: `rock`, `paper`, `scissors`
* Images resized to `224×224`
* Trained using cross-entropy loss
* Optimized with Adam optimizer
* Final Accuracy: 98.40%
* Final Loss: 6.167
* Final model saved as:

```text
rps_model_for_app.pth
```

---

## AI Behavior & Learning Strategy

The AI does **not simply react randomly**. It uses two layers of intelligence:

### 1. Computer Vision

* CNN predicts the player’s current hand gesture from webcam input

### 2. Behavioral Learning (Pattern Recognition)

* Tracks player move history
* Learns **move sequences**
* Predicts the most likely next move
* Counters predicted behavior

**Example:**
If a player frequently plays:

```
Scissors → Paper
Scissors → Paper
Scissors → Paper
```

The AI learns this pattern and counters accordingly.

This simulates **adaptive opponent behavior**, not just static rules.

---

## Performance Analysis & Results

### Gesture Recognition

* High accuracy under good lighting conditions
* Minor errors possible with:

  * Fast movement
  * Occluded hands
  * Poor lighting

### AI Adaptation

* Performs better as more rounds are played
* Becomes increasingly difficult to beat
* Successfully adapts to repetitive player strategies

### Real-Time Performance

* Runs smoothly at ~30 FPS
* No noticeable lag during gameplay
* Stable memory usage

---

## Limitations

* Requires a webcam
* Lighting conditions affect accuracy
* AI learning resets when the app closes

---

## Demo Video / Screenshots

### Demo Video

https://drive.google.com/file/d/1Y8xDz4s2OxqW7F1U4hpOLK-pV0YFKwVY/view?usp=sharing

### Screenshots

<img width="1387" height="732" alt="image" src="https://github.com/user-attachments/assets/4e5ae51f-f246-460b-9920-afd536e5f952" />
<img width="729" height="443" alt="image" src="https://github.com/user-attachments/assets/658fbb22-f3c3-422e-b440-bea8d86f7a18" />
<img width="870" height="850" alt="image" src="https://github.com/user-attachments/assets/779984f1-eb69-475c-85a7-b9b55a5d5e3d" />
<img width="832" height="872" alt="image" src="https://github.com/user-attachments/assets/6e207fc8-cb98-4b0a-a1d2-aba468c39334" />

---

## Conclusion

This project demonstrates:

* Practical application of deep learning
* Computer vision integration
* Real-time AI decision-making
* Desktop software deployment

It combines **machine learning theory** with **real-world usability**, resulting in an interactive and adaptive AI game.

---

## Credits

Developed by: *Mark Lourenco*
Course: *VGP 338*
Institution: *LaSalle College Vancouver*
