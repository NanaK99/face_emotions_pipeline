1. Run:
```
git clone https://github.com/MagicalLabs/face_emotions_pipeline.git
```
2. In the folder face_emotions_pipeline:
```
mkdir checkpoints
```
3. Download one of the pre-trained model weights to the *checkpoints* directory. Currently, weights named *AffectNet-8* are used, you can use any one from the available three optoins.

|     task    	| epochs 	| accuracy 	| link 	|
|:-----------:	|:------:	|:--------:	|:----:	|
| AffectNet-8 	|    5   	| 62.09    	|[download](https://drive.google.com/drive/u/0/folders/1HZlkkrgCiZXQqgj8XvsI3DK3kyorcKSp)      	|
| AffectNet-7 	|    6    	| 65.69     |[download](https://drive.google.com/drive/u/0/folders/1HZlkkrgCiZXQqgj8XvsI3DK3kyorcKSp)  
|    RAF-DB   	|   21   	| 89.70    	|[download](https://drive.google.com/drive/u/0/folders/1HZlkkrgCiZXQqgj8XvsI3DK3kyorcKSp)


4. In the face_emotions_pipeline directory, run: 
```
pip3 install -r requirements.txt
```
5. Run: 
```
python face_pipeline_with_emotions.py 
```

The latter command will read frames from the video path and output four different TextGrid files: 
- a file corresponding to the detected eye gaze movement, such as: "up left", "down right", "centre", etc,
- a file corresponding to the detected head or body movement, such as: "head shake", "head nod", "lean in or out",
- a file corresponding to the detected face expressions: such as, "Smiling", "Widened eyes", "Narrowed eyes", "Furrowed brows", "Tense" or "Raised Brows".
- a file corresponding to the detected emotion: such as, "happy", "sad", "anger", "fear", etc.
