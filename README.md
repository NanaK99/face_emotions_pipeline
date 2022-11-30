1. Run:
```
git clone https://github.com/MagicalLabs/face_emotions_pipeline.git
```
2. In the folder face_emotions_pipeline:
```
mkdir checkpoints
```
3. Download the below pre-trained model weights to the *checkpoints* directory.

|     task     	| link 	|
|:-----------:	|:------:	|
| AffectNet-8 	|[download](https://drive.google.com/drive/u/0/folders/1HZlkkrgCiZXQqgj8XvsI3DK3kyorcKSp)      	|

4. In the directory *static*, modify the file *config.ini* as you wish.

5. In the face_emotions_pipeline directory, run: 
```
pip3 install -r requirements.txt
```
6. Run: 
```
python face_pipeline.py --video video.mp4 --input_textgrid input.TextGrid --output_dir_name outputs
```
, where the first argument is the path to the video file, 
the second argument is the path to the input textgrid (which will be used to generate the output textgrids), and 
the third argument is a name of an output folder, where you wish to save the generated textgrids.

The latter command will read frames from the video path and output four different TextGrid files: 
- a file corresponding to the detected eye gaze movement, such as: "up left", "down right", "centre", etc,
- a file corresponding to the detected head or body movement, such as: "head shake", "head nod", "lean in or out",
- a file corresponding to the detected face expressions: such as, "Smiling", "Widened eyes", "Narrowed eyes", "Furrowed brows", "Tense" or "Raised Brows".
- a file corresponding to the detected emotion: such as, "happy", "sad", "anger", "fear", etc.
