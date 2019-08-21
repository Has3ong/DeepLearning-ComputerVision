## Convolutional Neural Network 1

### Train MNIST Data

Model
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               1179776   
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_________________________________________________________________
```


Result
```
Test loss: 0.03007729491974369
Test accuracy: 0.9906
```

fail case

predict data 7, validate data 9

<img width=500 src="https://user-images.githubusercontent.com/44635266/63226711-6752e400-c218-11e9-9fdd-a61daf60c697.png">

predict data 4, validate data 9

<img width=500 src="https://user-images.githubusercontent.com/44635266/63226712-7174e280-c218-11e9-827b-417ce5874b86.png">

predict data 0, validate data 6

<img width=500 src="https://user-images.githubusercontent.com/44635266/63226714-7cc80e00-c218-11e9-9c7a-19bb80ed51bf.png">