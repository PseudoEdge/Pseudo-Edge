## Running the Code
- Create the virtual environment
```
virtualenv pevenv --python=python3.8
. pevenv/bin/activate
```
- Install required packages
```
pip install -r requirements.txt
```
- Create following folders to store models:
```
mkdir best_model
mkdir best_pl_model
```
- Download edge candidates file and best pesudo-labels folder [here](https://drive.google.com/drive/folders/1QMc78I-w1rWg6Gw2y1Yzby8WIKFnXTgh?usp=sharing). Place the downloaded materials along with other files.

- To reproduce Pseudo-Edge w/GCN performance on OGBL-COLLAB, run the following script:
```
bash run.sh
```
