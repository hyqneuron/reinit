
## Comments

Several interesting observations:

1. VGG11 trained on cifar10 has lots of 0-units in the later stages. On the other hand, VGG11 pretrained on ImageNet has
   not a single 0-unit.
2. When data is lacking in complexity, large width will result in 0-units (cifar10). When data has enough complexity,
   large width can be fully filled.
3. `model3` uses Dropout2d. It removes 0-units better than vanilla Dropout, but still leaves a nontrivial number of
   0-units behind. This seems to suggest that even Dropout2d cannot fully eliminate 0-units. Of course, it could have
   been that I did not use strong enough dropout in `model3`. In `model4` I'll add in even stronger Dropout2d to find
   out if Dropout2d can indeed remove 0-units.
4. Even though Dropout2d reduces the number of 0-units, test accuracy stays about the same.
5. Stronger weight decay from 0.00005 to 0.0005 very marginally reduces test accuracy
6. One vgg16 trained with SGD has not one 0-unit. However, once I changed the optimizer to Adam the 0-units came back
   out. This is very likely due to Adam's update normalization. Units that are almost dead would get very small
   backpropagated errors. Consequently, the dominant update would come from weight decay. Since update normalization
   would scale a small weight decay update to a much larger one, it drives the weight to 0 much faster than SGD. This is
   verified in RMSprop, which shows a similarly large number of 0-units, and which also performs update normalization.

## model1
```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.00005
test_accuracy = 0.85
model_conf = [
    ('conv1_1', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, None)),
    ('conv3_2', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (512, None)),
    ('conv4_2', (512, None)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (512, None)),
    ('conv5_2', (512, None)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (512, 0.5)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (3,   64),
 'conv3_1/0': (6,   128),
 'conv3_2/0': (19,  256),
 'conv4_1/0': (31,  256),
 'conv4_2/0': (136, 512),
 'conv5_1/0': (241, 512),
 'conv5_2/0': (256, 512),
 'fc6/0':     (251, 512),
 'logit':     (2,   512)}

```

## model2

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.00005
test_accuracy = 0.846
model_conf = [
    ('conv1_1', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, None)),
    ('conv3_2', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (512, 0.5)),
    ('conv4_2', (512, 0.5)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (512, 0.5)),
    ('conv5_2', (512, 0.5)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (512, 0.5)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (1,   64),
 'conv3_1/0': (12,  128),
 'conv3_2/0': (26,  256),
 'conv4_1/0': (34,  256),
 'conv4_2/0': (108, 512),
 'conv5_1/0': (209, 512),
 'conv5_2/0': (112, 512),
 'fc6/0'    : (48,  512),
 'logit'    : (3,   512)}
```

## model3
  
model3 is trained with Dropout2d. Previous models weren't. Interestingly this didn't improve performance in any way. I
forgot to save model3.

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.00005
test_accuracy = 0.846
model_conf = [
    ('conv1_1', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, None)),
    ('conv3_2', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (512, 0.5)),
    ('conv4_2', (512, 0.5)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (512, 0.5)),
    ('conv5_2', (512, 0.5)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (512, 0.5)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (5,   64),
 'conv3_1/0': (11,  128),
 'conv3_2/0': (27,  256),
 'conv4_1/0': (33,  256),
 'conv4_2/0': (55,  512),
 'conv5_1/0': (100, 512),
 'conv5_2/0': (171, 512),
 'fc6/0'    : (17,  512),
 'logit'    : (3,   512)}
```

## model4

model4 is trained with all dropout removed. Still retained the same level of test accuracy. Apparently dropout doesn't
really impact the test accuracy from model1 to model4.

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.00005
test_accuracy = 0.847
model_conf = [
    ('conv1_1', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, None)),
    ('conv3_2', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (512, None)),
    ('conv4_2', (512, None)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (512, None)),
    ('conv5_2', (512, None)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (512, None)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (2,   64),
 'conv3_1/0': (4,   128),
 'conv3_2/0': (36,  256),
 'conv4_1/0': (34,  256),
 'conv4_2/0': (160, 512),
 'conv5_1/0': (233, 512),
 'conv5_2/0': (278, 512),
 'fc6/0'    : (270, 512),
 'logit'    : (32,  512)}
```

## model5

model5 is similar to model4, but with weight decay increased from 0.00005 to 0.0005. Surprisingly 0.00005 weight decay
was actually suprior. Apparently strong weight decay creates more dead units. Also surprising is how many dead units
there are and the net still works.

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.0005
test_accuracy = 0.837
model_conf = same as model4
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (15,  64),
 'conv3_1/0': (33,  128),
 'conv3_2/0': (92,  256),
 'conv4_1/0': (106, 256),
 'conv4_2/0': (329, 512),
 'conv5_1/0': (349, 512),
 'conv5_2/0': (311, 512),
 'fc6/0'    : (462, 512),
 'logit'    : (116, 512)}
```

## model6

model6 is similar to model5. Only thing changed is weight initialization. Switched to
huva.init_weights(use_in_channel=False), which brought the initial weight norm from about 200 to about 400, and
significantly reduced the number of dead units compared to model5.

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.0005
test_accuracy = 0.851
model_conf = same as model4
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (0,   64),
 'conv3_1/0': (0,   128),
 'conv3_2/0': (17,  256),
 'conv4_1/0': (14,  256),
 'conv4_2/0': (297, 512),
 'conv5_1/0': (378, 512),
 'conv5_2/0': (344, 512),
 'fc6/0'    : (382, 512),
 'logit'    : (0,   512)}
```

## model7

model7 is really a mistake. Switched to huva.init_weights(use_in_channel=True), while there was a bug in init_weights,
leading the n to become very small, thus weights very large. Initial weight norm was >700.

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.0005
test_accuracy = 0.851
model_conf = same as model4
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (0,   64),
 'conv3_1/0': (0,   128),
 'conv3_2/0': (7,   256),
 'conv4_1/0': (24,  256),
 'conv4_2/0': (313, 512),
 'conv5_1/0': (370, 512),
 'conv5_2/0': (339, 512),
 'fc6/0'    : (398, 512),
 'logit'    : (0,   512)}
```

## model8

model8 is similar to model6, with properly implemented huva.init_weights(use_in_channel=True)

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.0005
test_accuracy = 0.844
model_conf = same as model4
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (0,   64),
 'conv3_1/0': (0,   128),
 'conv3_2/0': (12,  256),
 'conv4_1/0': (21,  256),
 'conv4_2/0': (306, 512),
 'conv5_1/0': (364, 512),
 'conv5_2/0': (365, 512),
 'fc6/0'    : (376, 512),
 'logit'    : (0,   512)}
```

## model9

model9 uses maximum layer width of 256. Test accuracy same as bigger models, yet still about half of those later units
are 0-units.

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.0005
test_accuracy = 0.849
model_conf = [
    ('conv1_1', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, None)),
    ('conv3_2', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (256, None)),
    ('conv4_2', (256, None)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (256, None)),
    ('conv5_2', (256, None)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (256, None)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (0,   64),
 'conv3_1/0': (0,   128),
 'conv3_2/0': (11,  256),
 'conv4_1/0': (24,  256),
 'conv4_2/0': (65,  256),
 'conv5_1/0': (103, 256),
 'conv5_2/0': (127, 256),
 'fc6/0'    : (104, 256),
 'logit'    : (0,   256)}
```

## model10

model10 is still smaller than model9, and still achieves about the same test accuracy

```python
optimizer     = Adam
lr            = 20 0.001 and 5 0.0001
weight_decay  = 0.0005
test_accuracy = 0.842
model_conf = [
    ('conv1_1', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, None)),
    ('conv3_2', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (256, None)),
    ('conv4_2', (256, None)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (160, None)),
    ('conv5_2', (160, None)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (160, None)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]
computed_utilization = 
{'conv1_1/0': (0,  3),
 'conv2_1/0': (0,  64),
 'conv3_1/0': (0,  128),
 'conv3_2/0': (25, 256),
 'conv4_1/0': (18, 256),
 'conv4_2/0': (57, 256),
 'conv5_1/0': (112,256),
 'conv5_2/0': (62, 160),
 'fc6/0'    : (34, 160),
 'logit'    : (0,  160)}
```

## model11

model11 uses random horizontal flipping. Test accuracy increased as a result. Number of 0-units larger than model10
probably because of longer training time.

```python
optimizer     = Adam
lr            = 25 0.001 and 10 0.0001, 5 0.00001
weight_decay  = 0.0005
test_accuracy = 0.876
model_conf = [
    ('conv1_1', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, None)),
    ('conv3_2', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (256, None)),
    ('conv4_2', (256, None)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (160, None)),
    ('conv5_2', (160, None)),
    ('pool5'  , (2,   2)),
    ('fc6'    , (160, None)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv2_1/0': (6,   64),
 'conv3_1/0': (20,  128),
 'conv3_2/0': (80,  256),
 'conv4_1/0': (90,  256),
 'conv4_2/0': (119, 256),
 'conv5_1/0': (152, 256),
 'conv5_2/0': (114, 160),
 'fc6/0'    : (113, 160),
 'logit'    : (7,   160)}
```

## model12

model12 uses vgg16 and Adam. Lots of dead units.

```python
optimizer     = Adam
lr            = 25 0.001 and 25 0.0005, 10 0.00025, 10 0.000125, 10 0.0000625, 10 0.00003125
weight_decay  = 0.0005
test_accuracy = 0.905
model_conf = [
    ('conv1_1', (64,  0.3)),
    ('conv1_2', (64,  None)),
    ('pool1'  , (2,   2)),
    ('conv2_1', (128, 0.4)),
    ('conv2_2', (128, None)),
    ('pool2'  , (2,   2)),
    ('conv3_1', (256, 0.4)),
    ('conv3_2', (256, 0.4)),
    ('conv3_3', (256, None)),
    ('pool3'  , (2,   2)),
    ('conv4_1', (512, 0.4)),
    ('conv4_2', (512, 0.4)),
    ('conv4_3', (512, None)),
    ('pool4'  , (2,   2)),
    ('conv5_1', (512, 0.4)),
    ('conv5_2', (512, 0.4)),
    ('conv5_3', (512, None)),
    ('pool5'  , (2,   2)),
    ('drop5'  , (None,0.5)),
    ('fc6'    , (512, 0.5)),
    ('logit'  , (None,None)),
    ('flatter', (None,None))
]
computed_utilization = 
{'conv1_1/0': (0,   3),
 'conv1_2/0': (0,   64),
 'conv2_1/0': (1,   64),
 'conv2_2/0': (0,   128),
 'conv3_1/0': (16,  128),
 'conv3_2/0': (0,   256),
 'conv3_3/0': (1,   256),
 'conv4_1/0': (167, 256),
 'conv4_2/0': (156, 512),
 'conv4_3/0': (237, 512),
 'conv5_1/0': (431, 512),
 'conv5_2/0': (233, 512),
 'conv5_3/0': (293, 512),
 'fc6/0'    : (13,  512),
 'logit'    : (0,   512)}
```

## model13

model13 is similar to model12 in every way other than optimizer switching to SGD. Obviously, Switching to SGD alone
removes all of the dead units, but test accuracy is not improved. It's actually slightly worse.

```python
optimizer     = SGD
lr            = 45 0.1 and 10 0.05, 10 0.025, 5 0.0125, 5 0.00625, 5 0.003125, 5 0.0015625, 5 0.00078125
weight_decay  = 0.0005
test_accuracy = 0.901
model_conf = same as model12
computed_utilization = 
{'conv1_1/0': (0, 3),
 'conv1_2/0': (0, 64),
 'conv2_1/0': (0, 64),
 'conv2_2/0': (0, 128),
 'conv3_1/0': (0, 128),
 'conv3_2/0': (0, 256),
 'conv3_3/0': (0, 256),
 'conv4_1/0': (0, 256),
 'conv4_2/0': (0, 512),
 'conv4_3/0': (0, 512),
 'conv5_1/0': (0, 512),
 'conv5_2/0': (0, 512),
 'conv5_3/0': (0, 512),
 'fc6/0'    : (0, 512),
 'logit'    : (0, 512)}
```

## model14

model14 changes optimizer to RMSprop. Just like Adam, it gives a lot of dead units

```python
optimizer     = RMSprop, momentum=0.9, centered=False
lr            = 20 0.001, 20 0.00025, 10 0.000125, 10 0.0000625, 10 0.00003125
weight_decay  = 0.0005
test_accuracy = 0.890, but can be improved by a better run
model_conf = same as model12
computed_utilization = 
{'conv1_1/0': (0, 3),
 'conv1_2/0': (0,   64),
 'conv2_1/0': (0,   64),
 'conv2_2/0': (0,   128),
 'conv3_1/0': (40,  128),
 'conv3_2/0': (2,   256),
 'conv3_3/0': (2,   256),
 'conv4_1/0': (188, 256),
 'conv4_2/0': (212, 512),
 'conv4_3/0': (236, 512),
 'conv5_1/0': (463, 512),
 'conv5_2/0': (220, 512),
 'conv5_3/0': (286, 512),
 'fc6/0'    : (31,  512),
 'logit'    : (4,   512)}
```

## model15

model15 is similar to model12, with wd lowered to 0.0001. Test accuracy increased to 0.913. Did not save. With reduced
weight decay, overall weight norm is 1.5 larger than normal (400 vs 220). However, dead units were just as prevelant.

## model16

model15 is similar to model12, with wd set to zero. Weight norm kept growing, killed at norm=2000. Training accuracy
kept improving, but apparently overfitting was happening. When wd is zero, not a single dead unit was present. Also
interesting was the fact that when weight norm kept growing, it was as if effective learning rate kept dropping because
update norm was not affected (Adam normalizes it). Test accuracy, on the other hand, settled around 0.902

## model17

model17 is trained with Adam with separate weight decay, wd=0.1.

## model18

model18 is a 300-epoch full run of Adam.

```python
optimizer     = Adam
lr            = 0.001, [30]*10
weight_decay  = 0.0005
test_accuracy = 0.913
model_conf = same as model12
computed_utilization = 
{'conv1_1/0': (0, 3),
 'conv1_2/0': (0, 64),
 'conv2_1/0': (5, 64),
 'conv2_2/0': (0, 128),
 'conv3_1/0': (28, 128),
 'conv3_2/0': (0, 256),
 'conv3_3/0': (2, 256),
 'conv4_1/0': (135, 256),
 'conv4_2/0': (143, 512),
 'conv4_3/0': (226, 512),
 'conv5_1/0': (376, 512),
 'conv5_2/0': (245, 512),
 'conv5_3/0': (281, 512),
 'fc6/0': (12, 512),
 'logit': (1, 512)}
```

## model19

model19 is a 300-epoch full run of SGD.

```python
optimizer     = SGD
lr            = 0.001, [30]*10
weight_decay  = 0.0005
test_accuracy = 0.911
model_conf = same as model12
computed_utilization = 
{'conv1_1/0': (0, 3),
 'conv1_2/0': (0, 64),
 'conv2_1/0': (0, 64),
 'conv2_2/0': (0, 128),
 'conv3_1/0': (0, 128),
 'conv3_2/0': (0, 256),
 'conv3_3/0': (0, 256),
 'conv4_1/0': (0, 256),
 'conv4_2/0': (0, 512),
 'conv4_3/0': (0, 512),
 'conv5_1/0': (0, 512),
 'conv5_2/0': (0, 512),
 'conv5_3/0': (0, 512),
 'fc6/0': (0, 512),
 'logit': (0, 512)}
```

## model20

model20 is same as model18, with weight decay reduced to 0.0001. Test accuracy seems to be unaffected by the change in
weight decay. Compared to model18, this reduction in weight decay has not led to a reduction in the number of dead
units, showing that Adam indeed turns weight_decay irrelevant.

```python
optimizer     = Adam
lr            = 0.001, [30]*10
weight_decay  = 0.0001
test_accuracy = 0.912
model_conf = same as model12
computed_utilization = 
{'conv1_1/0': (0, 3),
 'conv1_2/0': (0, 64),
 'conv2_1/0': (0, 64),
 'conv2_2/0': (0, 128),
 'conv3_1/0': (0, 128),
 'conv3_2/0': (0, 256),
 'conv3_3/0': (0, 256),
 'conv4_1/0': (87, 256),
 'conv4_2/0': (43, 512),
 'conv4_3/0': (172, 512),
 'conv5_1/0': (476, 512),
 'conv5_2/0': (203, 512),
 'conv5_3/0': (284, 512),
 'fc6/0': (0, 512),
 'logit': (2, 512)}
```

## model21

model21 is same as model19, with weight decay reduced to 0.0001. Test accuracy seems to be unaffected by the change in
weight decay.
```python
optimizer     = SGD
lr            = 0.001, [30]*10
weight_decay  = 0.0001
test_accuracy = 0.911
model_conf = same as model12
```

## model22

Somehow all previous weight initialization were done with `n=k*k*num_in_channel`. I still have no idea why it went that
way. model22 is SGD, weight_decay=0.0005, similar to model19, but with weight initialization fixed to
`n=k*k*num_out_channel`

```python
optimizer     = SGD
lr            = 0.001, [30]*10
weight_decay  = 0.0005
test_accuracy = 
model_conf = same as model12
```
