# Facial Emotion Recognition

## Project Introduction 
This project aims to perform facial emotion recognition. I have used two different models:
* Keras MobileNet for facial emotion classification where I fined-tuned the base model and received an accuracy of 40%
* YOLO (You Only Look Once) for detecting face emotions in the input image or video. Here I receive an accuracy of 68%.

## Training

The models used in this project is trained on the [dataset](https://www.dropbox.com/s/nilt43hyl1dx82k/dataset.zip?dl=0), which contains facial emotion images labeled with seven different emotions (angry, disgust, fear, happy, sad, surprise, neutral). If you wish to train your own emotion classification model on a custom dataset, you can replace the pre-trained weights with your own trained model.


## Weights
* MobileNet Weight: [![Download](https://img.shields.io/badge/Download%20-8A2BE2)](https://drive.google.com/file/d/1ulbINKFjAti1NoaKvMBenA0ZuZmG7USZ/view?usp=drive_link)

* YOLO Weight: [![Download](https://img.shields.io/badge/Download%20-8A2BE2)](https://drive.google.com/file/d/1ulbINKFjAti1NoaKvMBenA0ZuZmG7USZ/view?usp=drive_link)
## Installation

To get started with the emotion recognition project, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Ketan3102/Emotion_Detection.git
   cd Emotion_Detection
2. Go to the model you would like to run.
3. Installation the required packages (preferably within a Virtual Environment):

    ```bash
    pip install -r requirements.txt
4. Download the YOLO model weights or MobileNet model weights and place them in the weights directory.


## Usage
1. To perform facial emotion recognition using the Keras MobileNet model, run:

    ```bash
    python emotion_detection.py
2.To perform facial emotion recognition using the YOLO model, run:

    ```bash
    python app.py
3. This will `open webcam` to detect your video and would tell you your emotions.
4. To stop the webcam or application, press `q` in keyboard.
