# EEG Motor Imagery Classification Using Deep Learning: #
## Introduction ##

This project explores how brain signals (EEG) can be used to recognize imagined movements like left hand, right hand, both feet, and tongue.
The work is based on the BCI Competition IV, Dataset 2a, where participants wore an EEG cap with 22 electrodes to record brain activity.

The goal of this project is to build deep learning models that can predict which movement a person is imagining just from their brain signals.

## What is EEG? ##

EEG (Electroencephalography) is a way to record the brain’s electrical activity using electrodes placed on the scalp.

Think of it like tiny microphones placed on your head that “listen” to your brain’s signals.
These signals resemble waves, and different wave patterns correspond to various mental activities, such as moving your hand, imagining a movement, or even sleeping.

## Dataset: ##

The dataset used here is BCI Competition IV, Dataset 2a.

Subjects: 9

Channels: 22 EEG electrodes

Classes (4 movements):

----> Left hand

----> Right hand

----> foot

----> Tongue

Epochs: Each recording is a 4-second clip of EEG data while the subject imagines a movement

## Models Used: ##

I experimented with several models and compared their performance:

CNN + LSTM: A combination of convolutional and recurrent networks.

GRU Models: GRU (Gated Recurrent Unit) networks for sequence learning.

Fast BiGRU + CNN (Best Model)

Bidirectional GRU: Reads brain signals forward and backward.

CNN: Detects local patterns in the signals.

This combination worked best because EEG data is both time-based (sequences) and spatial (channels across the scalp).

## What I Did: ##

Loaded and prepared the dataset.

Preprocessed signals (normalization, frequency features, sliding windows).

Created visualizations to understand the data:

Class and patient distributions

Raw vs normalized signals

Heatmaps and frequency plots

Accuracy/Loss training curves

Confusion matrix

Trained multiple models and compared their results.

Selected the Fast BiGRU + CNN model as the best performing.

## Results: ##

Best Test Accuracy: 96.9%

Test Loss: 0.3742

Classification Report: High precision, recall, and F1 across all 4 classes.

The model performed well after fine-tuning and using all training subjects.

## Challenges: ##

EEG data is noisy and complex.

Preprocessing was difficult compared to image or text datasets.

Model tuning required balancing underfitting and overfitting.

## Future Work: ##

Try other advanced models with attention mechanisms.

Explore real-time EEG classification.

Apply the model to more subjects and larger datasets.

## Applications: ##

This kind of EEG-based classification can be applied in:

Assistive technology: Brain-controlled wheelchairs or prosthetics.

Medical diagnosis: Seizures, sleep disorders, Alzheimer’s detection.

Mental health: Neurofeedback for stress, ADHD, and focus.

Gaming and VR: Brain-controlled games.

Human-Computer Interaction: Controlling devices without physical movement.

## How to Run the Project:

Clone this repository:

git clone ((https://github.com/AdeebaRafi/EEG_Classification_BCI))
cd EEG_Classification_BCI


## Install dependencies:

pip install -r requirements.txt


## Train the model:

python train.py

Evaluate the model and view results.

## Conclusion:

This project shows how deep learning and EEG data can be combined to classify imagined movements.
The results highlight the potential of brain-computer interfaces for real-world applications in healthcare, accessibility, and human–machine interaction.

