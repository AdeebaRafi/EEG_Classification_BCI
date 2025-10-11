# Detail about my Project for Future Reference

EEG Motor Imagery Classification Using Deep Learning.
The focus of this work is on Brain-Computer Interfaces, where I try to decode brain signals into meaningful actions.
Specifically, I built a model that can classify EEG signals recorded when a person imagines different movements like left or right hand, feet, or tongue.
The main goal is to explore how EEG, which records electrical activity from the brain, can be used for movement prediction in applications such as neurorehabilitation and assistive technology.

(Neurorehabilitation is a medical treatment that helps people recover after a brain injury or nervous system disease by improving their physical, cognitive, and emotional functions)

For this project, I worked on a 4-class motor imagery task.
The four classes represent the imagination of movements: left hand, right hand, feet, and tongue.
The dataset includes recordings from 9 subjects, each having multiple sessions.
This gives variety in the data but also introduces subject-to-subject differences, which makes classification more challenging.
The signals were recorded using 22 EEG channels that capture brain activity, along with 3 EOG channels that capture eye movements.
These help us detect and reduce noise from eye blinks or eye motion.

Since EEG is recorded as a time-series signal, the data represents how brain activity changes over time during each imagined movement.
EOG(Electrooculography) channels measure eye movements and blinks. They’re not for movement classes, but to catch eye-related noise.
This way we can clean the EEG signals, so the model learns brain activity only, not eye activity.

As a whole
I first cleaned the dataset by removing missing values and checking that all classes were balanced. Then for signal processing, I split the EEG into 50-timepoint windows using a sliding window with stride 10. 
Each channel was normalized, and I applied FFT to get frequency features. This way, I captured both time and frequency information. 
For feature engineering, the final input shape was 50 × 22 × 2, where 50 is time steps, 22 is channels, and 2 represents time and frequency features. Labels were then encoded into numbers so the model could process them."


# Data Preprocessing
• First, I cleaned the dataset. Any missing or bad values were removed so the data stays reliable. 
Then I checked that all four motor imagery tasks were balanced, meaning there were similar numbers of samples for each class.
This avoids bias in training.

# Signal Processing
EEG is a continuous time signal, so I cut it into smaller chunks using a sliding window. Each window had 50 timepoints, and I moved the window forward 10 points at a time. 
This way, I could capture overlapping temporal information instead of looking at one long signal.<img width="468" height="41" alt="image" src="https://github.com/user-attachments/assets/525953bc-6186-416d-8ad3-d40cca208afe" />

# Model Architecture & Selection 
For this project, I used a hybrid model that combines CNN and BiLSTM. The reason is that EEG signals have two important parts:
•	The spatial patterns, meaning how signals differ across electrodes on the scalp.
•	The temporal patterns, meaning how the signals change over time.
CNN is very good at extracting spatial features across channels, while BiLSTM is designed to capture time sequences. By combining both, the model can learn from both perspectives, making it more powerful.
I also added regularization methods — L2 and dropout — to avoid overfitting. Without this, the model might memorize the training data instead of learning general patterns.
Specifically, L2 regularization works by slightly punishing the model if any weight becomes too large. It keeps the weights small and balanced, so the model doesn’t depend too much on a few features. This helps it generalize better.
Finally, I used bidirectional LSTM, which looks not only at past signals but also at future context in the sequence. This is especially useful for EEG because brain signals are continuous, and understanding both directions gives a fuller picture.

# Model Design Philosophy
•	Model Used – CNN + BiLSTM hybrid:
I used a hybrid model that combines CNN and BiLSTM. CNN looks at the spatial features across EEG channels, and BiLSTM focuses on the time sequence of the signals.
•	Why hybrid?
EEG signals have two important parts — the patterns across channels (spatial) and how the signal changes over time (temporal). CNN is good for spatial, BiLSTM is good for temporal, so combining both makes the model stronger.
•	Regularization (L2 + dropout):
To avoid overfitting, I added L2 and dropout. These methods stop the model from memorizing the training data and help it perform well on new data.
Bidirectional processing (BiLSTM):
Normally, models only look at past data to predict the next step. BiLSTM looks at both past and future context in the signal, which helps capture more information from EEG.

What is L2 Regularization?
•	When training a model, sometimes it memorizes the training data too much (this is called overfitting). That makes the model perform badly on new, unseen data.
•	To avoid this, we use regularization.
•	L2 regularization means:
•	The model is slightly punished for having very large weights.
•	It adds a penalty = (sum of all weights squared) × (a small factor λ).
•	This pushes the model to keep weights small and balanced instead of over-relying on a few features.

# Training Strategy & Optimization 
For training, I used the Adam optimizer with a learning rate of 0.0005.
Adam updates weights step by step, and this small learning rate keeps learning stable.
The loss function is categorical crossentropy with label smoothing. This is suitable for multi-class problems, and label smoothing helps the model generalize instead of being overconfident.
I trained with a batch size of 32, meaning the model learns from 32 samples at a time instead of the full dataset.
Finally, I used a validation split of 20%, so the model is always tested on unseen data during training.

# Training Configuration
•	Optimizer: Adam with learning rate 0.0005
o	Adam is a popular algorithm that tells the model how to update weights to get better each step. Think of it like a smart teacher adjusting your learning speed.
o	Learning rate (0.0005) controls how big each adjustment step is.
	If it’s too big → the model may jump around and never learn.
	If it’s too small → the model learns very slowly.
	Here, 0.0005 is a gentle step size — stable, not too aggressive.
•	Loss Function: Categorical Crossentropy with label smoothing
o	Categorical Crossentropy is used when we classify data into more than 2 categories (like left hand, right hand, etc.).
o	It measures how far off the model’s predictions are from the true labels.
o	Label smoothing means: instead of telling the model “the correct answer is 100% class A, 0% others,” we say “it’s 90% class A, 10% others.”
	This avoids the model becoming too confident and helps generalize better.
•	Batch Size: 32 samples
o	The model doesn’t learn from the whole dataset at once.
o	Instead, it looks at 32 samples at a time, updates itself, then takes another 32, and so on.
o	This makes training faster and more memory-friendly.
•	Validation Split: 20%
o	Out of all your data, 20% is kept aside for checking the model’s performance (not used for training).
o	This helps see if the model works on data it hasn’t seen.

# Training Progress & Convergence 
Here I am showing how the model learned over time.
In the first 10 epochs(ee-poks), the accuracy went from chance level, about 34%, all the way up to around 90%. This was the fast learning stage where the model quickly picked up the main patterns in the EEG data.
Between epochs 11 and 25, the model already understood the basics, so the improvements slowed down. This stage was more about refining and stabilizing the predictions.
After epoch 26, the model was already very good, so I reduced the learning rate. At this point, it was making small, careful improvements — more like fine-tuning and polishing.
Now, if we look at the convergence metrics:
•	The final training accuracy was 99.8%, which means the model almost perfectly fit the training data.
•	The validation accuracy was 96.9%, which is also very high. This shows that the model generalizes well to unseen data and did not just memorize the training set.
•	Early stopping helped me stop training at the best point, so I avoided overfitting.
Finally, the learning curves showed healthy training: the model reached high accuracy quickly, the training and validation curves stayed close to each other, and techniques like dropout, L2 regularization, and label smoothing kept the model stable

# Model Evaluation & Insights
In model evaluation. The confusion matrix tells us how well the model distinguished between the classes. For example, right hand movements were classified with very high accuracy — 99% recall  while tongue movement was a bit more challenging at 93%. Still, confusion between classes was minimal, so the model separated them well.
In terms of feature importance, the GRU layers helped capture the temporal sequence in the EEG, the CNN extracted spatial features across channels, and FFT added frequency information. Together, these gave a richer understanding of the signals.
Finally, in real-world terms, this means the model can reliably detect motor imagery, which is promising for clinical use and brain-computer interfaces. Also, it adapts to different brain patterns, which is important since every person’s signals are slightly unique.


•	A CNN finds patterns across the EEG channels. It tells us which scalp areas are active.
•	An RNN reads the signal in time and remembers past events. It helps the model follow how the signal changes.
•	Spatial features are about where on the head the activity is. Temporal features are about how the activity changes over time.
•	Together, they let the model detect both where and when important brain signals happen.

# Visualization & Interpretability 
On this slide, I’m showing how I visualized the EEG data and interpreted the model.
First, I looked at the raw EEG signals — this is the original brain activity. Then I normalized them so channels are comparable. I also transformed the signals into frequency spectra to see dominant brain waves.
Next, I applied advanced analytics. Band power analysis told me how much theta, alpha, and beta activity was present, which relates to different mental states. PCA(Principal Component Analysis) helped me reduce the complex data into 2 dimensions, so I could visualize how well the classes separate. Heatmaps showed me when and where brain activity was strongest across the scalp.
For interpretability, I used attention mechanisms, which let the model highlight the most important time segments. Feature importance showed which frequency bands were critical. Finally, decision boundaries made it clear that the model could separate different tasks reliably.
  
PCA (Principal Component Analysis): Helps us see how well the classes (e.g., right hand vs tongue movement) separate in feature space.
  

