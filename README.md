1. Run 
```
pip3 install -r requirements.txt
```
2. Run 
```
python face_pipeline.py
```

The latter command will use your device's videocam and display three different outputs, side by side: 
- first, a text corresponding to your eye gaze movement, such as: "up left", "down right", "centre", etc,
- second, a text showing a detected movement, such as: head shake, head nod, lean in or out,
- third, a text corresponding to your face expressions: such as, "Smiling", "Widened eyes", "Narrowed eyes", "Furrowed brows", "Tense" or "Raised Brows".
