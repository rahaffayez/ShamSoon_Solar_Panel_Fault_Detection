# Faulty Solar Panel Detection

This repository contains the code and resources for my graduation project on **Faulty Solar Panel Detection**. The project involves two main tasks:
1. **Classification** of faulty solar panels using VGG16 and transfer learning.
2. **Detection** of faults in solar panels using YOLO (You Only Look Once).

The project is organized into three main directories: `Classification`, `Detection`, and `docs`.

---

## Project Overview

The goal of this project is to detect and classify faults in solar panels using deep learning techniques. The project is divided into two parts:
1. **Classification**: Using a pre-trained VGG16 model with transfer learning to classify solar panel images as faulty or non-faulty.
2. **Detection**: Using YOLO to detect and localize faults in solar panel images.

This project aims to improve the efficiency of solar panel maintenance by automating the process of fault detection and classification.

---

## Dataset

### Classification Dataset
The classification dataset is sourced from Kaggle:  
[Faulty Solar Panel Images on Kaggle](https://www.kaggle.com/datasets/kiyoshi732/faulty-solar-panel-images).  
This dataset contains labeled images of solar panels, categorized as Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered..

### Detection Dataset
The detection dataset is sourced from Roboflow:  
[Solar Panel Faulty Detection on Roboflow](https://universe.roboflow.com/6rainstorm-yqytq/solar-panel-faulty-detection).  
This dataset is annotated for object detection and is used to train the YOLO model.

---


