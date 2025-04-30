# NSCLC Classifier Model

## Guide
### Preprocessing
1. Head over to the preprocessing.py file and look for `dicom_path` and `anno_path` variables located in the main function
```commandline
# preprocessing.py
def main():
    dicom_path = Path(r'D:\Datasets\Dataset')    // CHANGE THE PATH ON WHERE THE DATASET IS STORED IN YOUR COMPUTER
    anno_path = Path(r'D:\Datasets\Annotation')  // DO THE SAME WITH THE ANNOTATIONS
    ...
```
2. Ensure that a directory named `data` is created in the project folder (outside the src package)
3. Run the script found at the very bottom of the file
```commandline
if __name__ == '__main__':
    main()
```
### Training
Before running the train.py file, you can tweak some values contained in the model.yml file located int the configs directory.
Here you can change most of the hyperparameters and parameters that the model uses such as the `learning rate`, `epochs`, `optimizer`, and many more.

After changing some values if necessary, you can now run the train.py file. A logger is implemented so that you can check training logs in the logs directory.
The log file will be created everytime you run the script wherein the logfile is named such that it stores the timestamp of when you first ran the program.

During training, after every epoch, the current state of the model will be saved, this is known as `checkpoints` to ensure that you can continue training on another day without loosing any progress.
Just keep in mind that you can only continue training as long as you don't modify anything in the model.yml and the scripts inside the model package as the model parameters need to be preserved
all throughout the epochs.