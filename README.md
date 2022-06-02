# Action_recognition_on_videos
Pedestrian action recognition on videos with single and multi stream architecture. 

# Prerequisite requirements

1. Pytorch-lightning - ```pip install pytorch-lightning```
2. Matplotlib - ``` pip install matplotlib```
3. Open Cv - ```pip install opencv-python```
4. Numpy - ```pip install numpy```
5. CSV file - ```pip install csvfile```
6. Pandas - ```pip install pandas```

Optical flow videos can be convetered using  ```utils/video_to_colorflow_convertor.py``` 

To reduce the processing time videos are stored as compressed numpy array in ```.npz``` format and be converted using ```utils/videos_numpy_converter.py``` script  

# Training
Action recognition has been trained with single stream network on using resnet2+1D and 3D-CNN and that can be found in the seperate folder. 

The inputs such as location of the dataset and other training configiration are edited in the config.yaml of the seperate folder

To train each architecute 

 for 3D CNN architecture ```python 3dcnn/train.py 3dcnn/config.yaml```
 
 for resnet 2+1 D architecture ```python r(2+1)d/train.py r(2+1)d/config.yaml```
 
 for two stream network ```python two_stream_network/train.py two_stream_network/config.yaml```
 
