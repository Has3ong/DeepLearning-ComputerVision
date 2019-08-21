
## CNN 6

### Classfier Fruit

### Reference

* https://www.researchgate.net/publication/321475443_Fruit_recognition_from_images_using_deep_learning

Model

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 30, 30, 32)        9248      
_________________________________________________________________
activation_2 (Activation)    (None, 30, 30, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 15, 64)        18496     
_________________________________________________________________
activation_3 (Activation)    (None, 15, 15, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 13, 13, 64)        36928     
_________________________________________________________________
activation_4 (Activation)    (None, 13, 13, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1180160   
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 81)                41553     
_________________________________________________________________
activation_6 (Activation)    (None, 81)                0         
=================================================================
Total params: 1,287,281
Trainable params: 1,287,281
Non-trainable params: 0
_________________________________________________________________
```

Result

```
  precision    recall  f1-score   support

     Apple Braeburn       0.51      0.55      0.53       164
     Apple Golden 1       0.64      0.68      0.66       164
     Apple Golden 2       0.65      0.68      0.66       164
     Apple Golden 3       0.57      0.45      0.50       161
 Apple Granny Smith       0.68      0.68      0.68       164
        Apple Red 1       0.53      0.68      0.59       164
        Apple Red 2       0.55      0.59      0.57       164
        Apple Red 3       0.80      0.35      0.49       144
Apple Red Delicious       0.68      0.68      0.68       166
   Apple Red Yellow       0.68      0.68      0.68       164
            Apricot       0.68      0.68      0.68       164
            Avocado       0.63      0.63      0.63       143
       Avocado ripe       0.67      0.64      0.65       166
             Banana       0.68      0.67      0.68       166
         Banana Red       0.60      0.49      0.54       166
       Cactus fruit       0.46      0.68      0.55       166
       Cantaloupe 1       0.68      0.68      0.68       164
       Cantaloupe 2       0.60      0.68      0.63       164
          Carambula       0.66      0.54      0.60       166
           Cherry 1       0.58      0.62      0.60       164
           Cherry 2       0.80      0.81      0.80       246
     Cherry Rainier       0.74      0.60      0.66       246
   Cherry Wax Black       0.67      0.67      0.67       164
     Cherry Wax Red       0.68      0.68      0.68       164
  Cherry Wax Yellow       0.68      0.68      0.68       164
         Clementine       0.68      0.68      0.68       166
              Cocos       0.68      0.68      0.68       166
              Dates       0.68      0.68      0.68       166
         Granadilla       0.68      0.68      0.68       166
         Grape Pink       0.45      0.68      0.54       164
        Grape White       0.68      0.68      0.68       166
      Grape White 2       0.68      0.68      0.68       166
    Grapefruit Pink       0.68      0.68      0.68       166
   Grapefruit White       0.65      0.68      0.66       164
              Guava       0.68      0.68      0.68       166
        Huckleberry       0.68      0.68      0.68       166
               Kaki       0.68      0.68      0.68       166
               Kiwi       0.66      0.66      0.66       156
           Kumquats       0.68      0.68      0.68       166
              Lemon       0.66      0.63      0.64       164
        Lemon Meyer       0.68      0.68      0.68       166
              Limes       0.68      0.68      0.68       166
             Lychee       0.68      0.68      0.68       166
          Mandarine       0.68      0.68      0.68       166
              Mango       0.68      0.68      0.68       166
           Maracuja       0.62      0.60      0.61       166
 Melon Piel de Sapo       0.78      0.78      0.78       246
           Mulberry       0.72      0.38      0.50       164
          Nectarine       0.63      0.67      0.65       164
             Orange       0.67      0.67      0.67       160
             Papaya       0.68      0.68      0.68       164
      Passion Fruit       0.65      0.68      0.67       166
              Peach       0.68      0.68      0.68       164
         Peach Flat       0.63      0.68      0.65       164
               Pear       0.56      0.68      0.61       164
         Pear Abate       0.78      0.66      0.71       166
       Pear Monster       0.63      0.68      0.66       166
      Pear Williams       0.65      0.58      0.61       166
             Pepino       0.66      0.68      0.67       166
           Physalis       0.68      0.68      0.68       164
 Physalis with Husk       0.65      0.67      0.66       164
          Pineapple       0.59      0.68      0.63       166
     Pineapple Mini       0.67      0.66      0.67       163
       Pitahaya Red       0.68      0.67      0.68       166
               Plum       0.63      0.65      0.64       151
        Pomegranate       0.38      0.25      0.30       164
             Quince       0.68      0.68      0.68       166
           Rambutan       0.68      0.68      0.68       164
          Raspberry       0.68      0.68      0.68       166
              Salak       0.56      0.67      0.61       162
         Strawberry       0.67      0.59      0.63       164
   Strawberry Wedge       0.78      0.77      0.78       246
          Tamarillo       0.68      0.68      0.68       166
            Tangelo       0.68      0.68      0.68       166
           Tomato 1       0.78      0.78      0.78       246
           Tomato 2       0.72      0.60      0.66       225
           Tomato 3       0.68      0.78      0.73       246
           Tomato 4       0.67      0.67      0.67       160
  Tomato Cherry Red       0.68      0.68      0.68       164
      Tomato Maroon       0.58      0.58      0.58       127
             Walnut       0.79      0.79      0.79       249

           accuracy                           0.66     13877
          macro avg       0.66      0.65      0.65     13877
       weighted avg       0.66      0.66      0.66     13877
```

