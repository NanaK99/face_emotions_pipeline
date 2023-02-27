1. Run:
```
git clone https://github.com/MagicalLabs/face_emotions_pipeline.git
```
2. In the folder face_emotions_pipeline:
```
mkdir checkpoints
```
3. Download the desired pre-trained model weights to the *checkpoints* directory. **It is recommended to use the weights of affectnet8.**

|     task     	| link 	|
|:-----------:	|:------:	|
| AffectNet-8 	|[download](https://drive.google.com/drive/folders/1K9zCWL9pBgNfAaQ26dAETcEBijMgyf8W?usp=share_link)      	|

4. In the directory *static*, modify the file *config.ini* as you wish.

5. In the face_emotions_pipeline directory, run: 
```
pip3 install -r requirements.txt
```
6. Run: 
```
python fp.py
```
This command has five optional arguments: --gaze, --body, --emotion, --expression, and --verbose. These arguments are set to False by default and to True if mentioned in the command line when executing the *python fp.py* command.

The latter command will read frames from the webcam and print the output texts corresponding to the arguments mentioned in the command line:
- GAZE: the detected eye gaze movement, such as: "up left", "down right", "centre", etc,
- BODY: the detected head or body movement, such as: "head shake", "head nod", "shoulder movement",
- EXPRESSION: the detected face expressions: such as, "Smiling", "Widened eyes", "Narrowed eyes", "Furrowed brows", "Tense" or "Raised Brows".
- EMOTION: the detected emotion: such as, "happy", "sad", "anger", "fear", etc.
