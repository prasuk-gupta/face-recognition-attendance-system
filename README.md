# AI-Based Face Recognition Attendance System

This repository contains a real-time face recognition attendance system developed
as an academic and practical project inspired by modern surveillance and
access-control pipelines.

The system uses **YOLOv8** for fast and accurate face detection and **FaceNet
(InceptionResnetV1)** for embedding-based face recognition.

---

## Overview

The application captures live video from a camera, detects faces in real time,
matches them against stored facial embeddings, and marks attendance only once per
day for each recognized individual. Attendance records are stored locally with
timestamps for tracking and verification.

This project focuses on the **end-to-end computer vision pipeline**, similar to
those used in smart surveillance and security systems.

---

## Key Features

- Real-time face detection using YOLOv8  
- Face recognition using FaceNet embeddings  
- Embedding-based identity matching with distance thresholding  
- Duplicate attendance prevention (per day)  
- Timestamp-based attendance logging  
- CPU/GPU compatible execution  

---

## Tech Stack

- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- FaceNet (InceptionResnetV1)  
- PyTorch  
- NumPy  

---

## Project Structure
.
├── detection/
│ ├── config.yaml
│ ├── yolov8_detector.py
│ └── yolov8_trainer.py
├── face_recognition.py
├── generate_face_embeddings.py
└── README.md


---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt

