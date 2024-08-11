
# DDoS Attack Detection and Mitigation in SDN Using ML Algorithms

This project focuses on developing a system for detecting and mitigating Distributed Denial of Service (DDoS) attacks in Software-Defined Networking (SDN) environments using machine learning algorithms.

## Introduction

DDoS attacks are one of the most prevalent security threats to modern networks. In SDN, the separation of the control plane from the data plane offers unique opportunities for deploying intelligent detection and mitigation systems. This project leverages machine learning algorithms to identify and mitigate DDoS attacks in real-time.

## Features

- **Real-time DDoS Detection:** Utilizes decision tree (DT), RF and k-nearest neighbors (KNN) algorithms for accurate detection.
- **Efficient Mitigation:** Implements mitigation strategies to allow legitimate traffic without delay.
- **Scalability:** Designed to work in various SDN environments.
- **Modular Design:** Easily extendable for adding more ML algorithms.

## Architecture

The system architecture includes:

1. **SDN Controller (Ryu):** Manages the flow of network traffic.
2. **DDoS Detection Module:** Uses trained ML models to classify network traffic.
3. **Mitigation Module:** Enforces mitigation strategies based on detection results.

## Machine Learning Algorithms

- **Decision Tree (DT):** Provides high accuracy in identifying malicious traffic.
- **K-Nearest Neighbors (KNN):** A simple yet effective algorithm for detecting anomalies in network traffic.

## Dataset

The dataset used for training and testing the models was generated using Mininet. It includes a mix of legitimate and malicious traffic to ensure comprehensive coverage.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thesaajii/Ddos-attack-detection-and-mitigation-using-ML.git
   ```
2. **Install the required dependencies:**
   ```bash
   ubuntu 20 or below
   Mininet
   Ryu controller
   
   ```

## Usage

1. **Start the Ryu controller:**
   ```bash
   ryu-manager controller_name .py
   ```
2. **Run the Mininet simulation:**
   ```bash
   sudo python3 topology.py
   ```
## hping commands
# icmp flood
hping3 -1 -V -d 120 -w 64 -p 80 --rand-source --flood
# syn flood
hping3 -S -V -d 120 -w 64 -p 80 --rand-source --flood
# udp flood
hping3 -2 -V -d 120 -w 64 -p 80 --rand-source --flood

## Results

The system successfully detects and mitigates DDoS attacks in real-time with minimal impact on legitimate traffic. Detailed performance metrics and evaluation results can be found in the `results` directory.

## License

This project is licensed 
