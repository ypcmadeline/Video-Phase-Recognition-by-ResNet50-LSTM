# Video-Phase-Recognition-by-ResNet50-LSTM
A temporal neural network is implemented to recognize different phases in sugurical video. 

## Data
The Cholec80 dataset is used in this project. The videos are annotated with phases. The videos are sampled at 100 fps. 

## Model
A model of ResNet50 + LSTM is used to process the images. <br />
![image](https://github.com/ypcmadeline/Video-Phase-Recognition-by-ResNet50-LSTM/blob/main/media/model.png)
ResNet50 is used to extract features from 3 frames and the 3 feature vectors are passed to the LSTM to learn the temporal relationship between the frames.

## Result
The accuracy and loss plot. <br />
![image](https://github.com/ypcmadeline/Video-Phase-Recognition-by-ResNet50-LSTM/blob/main/media/result.png)
