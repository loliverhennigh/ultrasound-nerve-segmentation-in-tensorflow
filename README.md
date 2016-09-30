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
Once the model is trained suffiecently you can evaluate it by running
```
python nerve_test.py
```
This will generated the file to be evalutated by the kaggle website. (hold on, need to push from different computer tomorrow)


