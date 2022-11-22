1. Run:
```
git clone https://github.com/MagicalLabs/face_emotions_pipeline.git
```
2. In the folder face_emotions_pipeline, run:
```
git clone https://github.com/yaoing/DAN.git
```
3. Create a new directory, named *checkpoints*, in the DAN directory and from the repo (of the 2nd step) download the pre-trained model weights. Currently, weights named *AffectNet-8* are used, you can use any one from the available three optoins.

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
