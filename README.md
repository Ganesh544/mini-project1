# mini-project1
Adaptive Congestion Detection and Traffic Control in SDN
# Intelligent Traffic Control and Attack Detection in Software-Defined Networks

## Project Overview
This project presents an intelligent traffic management framework for Software-Defined Networking (SDN) environments using the OpenFlow protocol and machine learning techniques. The system is designed to monitor network traffic, identify abnormal behavior, classify attack types, and predict network throughput for efficient traffic control and resource optimization.

By combining centralized SDN control with data-driven analytics, the project enhances network security, scalability, and overall performance. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Key Features
- Real-time network traffic analysis  
- Attack detection and classification  
- Throughput prediction using machine learning  
- Automated data preprocessing and feature engineering  
- Interactive web-based dashboard  
- Model performance visualization  
- Batch and single-instance prediction support  

## Technical Stack
- Python  
- Flask Framework    
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  

## Machine Learning Models
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Tao Tree Ensemble Model  

## System Workflow
1. Traffic dataset collection  
2. Data preprocessing and balancing  
3. Feature extraction and scaling  
4. Model training and evaluation  
5. Attack classification  
6. Throughput prediction  
7. Intelligent traffic management  

## Project Objectives
- Improve network traffic efficiency  
- Detect malicious or abnormal traffic patterns  
- Reduce congestion in SDN environments  
- Enable predictive and adaptive traffic control  
- Support secure and scalable network management  

## Application Execution

```bash
pip install -r requirements.txt
python app.py
```

The application runs locally on:

```text
http://localhost:5000
```

## Project Structure
- `app.py` — Web application and routing :contentReference[oaicite:4]{index=4}  
- `ml_models.py` — Model training, evaluation, and prediction engine :contentReference[oaicite:5]{index=5}  
- `utils.py` — Utility functions and file validation :contentReference[oaicite:6]{index=6}  

## Future Scope
- Integration with live SDN controllers  
- Real-time packet monitoring  
- Cloud-based deployment  
- Advanced anomaly detection using deep learning  

## Author
Ganesh Kadari
