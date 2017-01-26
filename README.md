# ultrasound-nerve-segmentation-in-tensorflow

I took a wack at the [ultrasound nerve segmentation challange](https://www.kaggle.com/c/ultrasound-nerve-segmentation) with tensorflow. This challenge is to accurately identify nerve structures in ultrasound images. 

Jumping in really late to the kaggle ultrasound nerve segmentation competition. Hopefully save peoples time in writing boring data loading scripts.

# Getting data and making TFrecords
To get the data, download the `test.zip` and `train.zip` files from the kaggle website listed above. Place these in the data directory and unzip them making two directories `test` and `train`. Now to generate the tf records enter the `utils` file and run
```
python createTFRecords.py
```
This will take about a minute and generate a 2.7 gigibyte file containing the train datasets in tfrecord form.

# Training
To train enter the `train` directory and run
```
python nerve_train.py
```
This has a default of set to train for 500,000 steps which is about 100 epochs.

# Tensorboard
Some training information such as the loss is recorded and can be viewed with tensorboard. The checkpoint file is found in `checkpoint` and has a default name `train_store_run_0001`.

# Evaulation
Once the model is trained sufficiently you can evaluate it by running
```
python nerve_test.py
```
This will generated run length encoding text file that is needed for kaggle to check. You can also run this with the `--view_images=True` flag to display the predicted 

# Model details
This network is a typical U-net with residual layers and 4 down samples. The residual layer have the option for different nonlinearitys and whether they are gated or not. The default is [Concatenated Elu](https://arxiv.org/pdf/1603.05201.pdf) for the activation function and gated set to True. More details can be found in the `nerve_architecture.py`. The network is very similar to that seen in [PixelCNN++](https://openreview.net/pdf?id=BJrFC6ceg).

# Performance
This network was able to get .61230 accuracy on the test set. This is fairly impressive because similar U-nets that I have seen are only able to get around .57 [here](https://github.com/jocicmarko/ultrasound-nerve-segmentation). In order to get in the .65 - .69 accuracy range data augmentation seems to be necessary like seen [here](https://github.com/EdwardTyantov/ultrasound-nerve-segmentation).


