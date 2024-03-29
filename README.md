# Apparel Recommender using Deep Learning

This project utilizes transfer learning with TensorFlow's ResNet50 model to build an apparel recommendation system. The system scans images and displays the five closest items from the dataset on a Streamlit-based webpage.
**Could not host the project due to the dataset of images being extremely large**
## Overview

The goal of this project is to leverage the pre-trained ResNet50 model to extract features from apparel images and recommend visually similar items from the dataset. The process involves:

- Utilizing TensorFlow and Keras for implementing transfer learning with ResNet50.
- Building a Streamlit-based web application for user interaction and display.

## Features

- Transfer learning with ResNet50 for image feature extraction.
- Displaying the five closest apparel items based on similarity.
- Streamlit-based interface for user interaction.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ayushtiwari134/apparel_recommender_dl.git
    ```
    
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the app locally using Streamlit:
```bash
streamlit run app.py
```
## Demo
1. Option to upload a file
<img src="/demo/Screenshot 2024-01-07 235800.png" alt="Alt text" title="Optional title">

2. Uploading a file and displaying the uploaded file
<img src="/demo/Screenshot 2024-01-07 235944.png" alt="Alt text" title="Optional title">

3. The nearest recommendations from the dataset of 10,000 images are shown below the uploaded file.
<img src="/demo/Screenshot 2024-01-08 000050.png" alt="Alt text" title="Optional title">


