# An autonomous truck simulator for the game American Truck Simulator

## Data

VOCdevkit2 2007:

http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

### train3/ directory

- this directory is created by the user with `record_frames.py`
- to get an adequate amount of data the game must be played for several hours

### labels_directions.csv

- this file is generated after training data is collected and corresponds to the individual frames saved from the training data indicating a `left`, `right`, or `straight`

