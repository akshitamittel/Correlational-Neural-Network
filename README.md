#Reconstructing images using Correlational Neural Network

Augmented Correlational Neural Networks to reconstruct images in MNIST number dataset as well as more complex images present in the CIFAR10 dataset.

Project by: Akshita Mittel (cs13b1040)

##CorrNet

Copy the cifar10 folder into each of the mnistExample directories in each module

###CorrNet Master:
>MNIST basic example
>CIFAR10 basic example:
>CorrNet 5-layered
>CorrNet-14

To create the dataset for training, run the following command, in the mnistExample directory for each module:
```
$ python create_data.py MNIST_DIR/ (or any directory containing your training data)
```

Next, to train the CorrNet, run the following command.
```
$ python train_corrnet.py MNIST_DIR/ TGT_DIR/ (Target directory)
```

To project the data to the learnt space, run the following command.
```
$ python project_corrnet.py MNIST_DIR/ TGT_DIR/
```

To reconstruct the data  run the following command:
```
$ python reconstruct.py MNIST_DIR/ TGT_DIR/
```
copy the copy_image.py code from this directory and paste it into the newly created reconstruct folder.

Then in the target directory, redirect to the reconstuct directory, type in the following:
```
$python create_image.py
```


To evaluate the learnt model for transfer learning task, run the following command:
```
$ python evaluate.py tl TGT_DIR/
```

To compute sum correlation in the projected space, run the following command.
```
$ python evaluate.py corr TGT_DIR/
```

###All references used in the project are mentioned in the last page of the document named presentation.
