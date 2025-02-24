![Logo](/web/public/Logo.png)

# AI-Driven Efficient MRI Super-Resolution

## Overview

This project aims to revolutionize MRI imaging technology by leveraging advanced AI techniques to enhance image quality, reduce scanning times, and lower costs. By upgrading MRI systems from 1.5T to 3T, we strive to improve diagnostic accuracy and patient outcomes in healthcare settings.

## 🌟 Inspiration

MRI scans are inherently expensive, and high-quality **3T models** escalate costs even further, creating significant barriers to access in many regions. Our goal was to leverage **AI-driven image processing** to enhance accessibility, particularly in **resource-limited settings**, making advanced medical imaging more **affordable** and **widely available** to improve **patient outcomes** and **diagnostic capabilities**.

## Features

- **Machine Learning Model:** Developed to provide superior image resolution and quality.
- **Efficient Image Processing:** A dedicated backend for handling image data, ensuring quick and reliable processing.
- **User-Friendly Frontend:** An intuitive UI that integrates seamlessly with backend services for enhanced user experience.
- **ML Parameter Testing:** Rigorous testing to optimize the machine learning model's performance.

## ⚠️ Challenges We Ran Into

We faced several challenges, including:

- **Lack of pair images** 🏥  
  We didn't have a pair of images to validate the generated images against.

- **Managing computational demands** 💻  
  Optimizing our model to efficiently leverage available resources.  

- **Balancing realism and medical accuracy** 🏥  
  Ensuring the generated images are **clinically useful** for diagnosis.  

## Team

- **Team Member 1:** Machine Learning Developer  
  Developed the machine learning model that powers our AI-driven MRI solution.

- **Team Member 2:** ML Backend Developer  
  Built the backend infrastructure for image processing, ensuring efficient data handling.

- **Team Member 3:** Frontend Developer  
  Created the frontend UI with seamless integration of backend services for optimal user experience.

- **Team Member 4:** ML Parameter Tester  
  Conducted extensive parameter testing to refine the machine learning models and enhance performance.

## Use Cases

Our AI-driven solution benefits various stakeholders in the healthcare industry:

- **Radiologists:** Enhanced image clarity and detail for more accurate diagnoses.
- **Healthcare Administrators:** Improved operational efficiency with reduced scanning times and lower costs.
- **Patients:** Quicker and more accurate diagnoses leading to faster treatment plans.
- **Technologists:** Advanced AI tools to streamline workflows and enhance the overall scanning process.

## Details

🚀 **Tech Stack:**  
🟢 **Framework:** Python, PyTorch Lightning, plotlib   
🟢 **Techniques Used:** Contrastive Unpaired Translation

## Demo Video
- Link  to a working demo of our project. [MRI_Video](https://www.youtube.com/watch?v=W0isckv7zg4)


## Installation/Run Reminders
- To run the frontend, enter the `web` directory and use:
  ```bash
  npm install
- To run the backend
  ```bash
  pip install -r requirements.txt
- To run the server, enter the `backend` directory 
  ```bash
  fastapi dev fast.py
  
## References:
These Projects and Papers were used as references for this project:
- https://github.com/wilbertcaine/CUT
- https://github.com/xavysp/TEED
- https://arxiv.org/abs/1703.10593
- https://ieeexplore.ieee.org/document/9622180
- https://github.com/taesungp/contrastive-unpaired-translation/tree/master

## Dataset
- Dataset for this usecase was obtained from: [MRI_Dataset](http://brain-development.org/ixi-dataset/)
