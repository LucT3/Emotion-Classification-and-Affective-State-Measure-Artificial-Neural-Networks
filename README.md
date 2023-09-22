# Emotion Classification and Affective State Measure Atificial Neural Networks
Intelligent Systems course project (MSc in Computer Engineering @ Unversity of Pisa). Design and development of a Multi Layer Perceptron (MLP), Radial Basis Function (RBF) networks and a Fuzzy System to estimate person's affective state. Design and development of a CNN to classify person's emotions.
</br></br>

## Table of Contents
1. [Technologies and Programming Languages](#technologies-and-programming-languages)
2. [Project Specifications](#project-specifications)
3. [Project Contents](#project-contents)
</br></br>

## Technologies and Programming Languages
<div align="center">
	<img width="50" src="https://user-images.githubusercontent.com/25181517/192106593-610ee31c-995e-4f24-b8e1-0f18eead6fae.png" alt="MATLAB" title="MATLAB"/>
	<br />
 	<img src="https://img.shields.io/badge/Matlab-FF6F00?style=for-the-badge&logo=tensorflow=white"/>
</div>


## Project Specifications
The aim of the first part of this project is to design and develop an intelligent system that measures a person’s affective state based on various biomedical signals (ECG, EEG, GSR) that are recorded by sensors.

The purpose of the second part is the classification of emotions on the basis of images depicting faces of people with different facial expressions related to a given emotion.
</br></br>

## Project Contents

+ **Datasets:**
  * Dataset with Biomedical Signals (Used for tasks 3.1 and 3.3)
  * Dataset of Images (Used for tasks 4.1 and 4.2)

+ **MLP (MultiLayer Perceptron) and RBF (Radial Basis Function) Neural Networks:**
  * Neural Networks used to estimate person's emotions (TASK 3.1).
  * Data preprocessing and data augmentation provided

+ **Mamdani Fuzzy Inference System:**
  * FIS that fix the deficiencies of estimate person's emotions only with NN (TASK 3.3)
  * Developed by using both Fuzzy Logic Designer and Matlab command line.

+ **CNN (Convolutional Neural Networks):**
  * Deep Neural Networks used to classify person's facial expressions (TASK 4.1 and 4.2).
  * 2-class (anger and happiness) and 4-class (anger,disgust, fear and happiness) classification solutions
  * Two different CNNs used:
    * Built from scratch CNN
    * Fine-tune of a pretrained CNN (AlexNet)

+ **Project Report**
