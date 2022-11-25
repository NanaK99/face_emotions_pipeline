from emotion_detection.dan import DAN
from configparser import ConfigParser
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import torch
import cv2


config_object = ConfigParser()
config_object.read("./static/config.ini")

checkpoints = config_object["CHECKPOINTS"]
checkpoints_path = checkpoints["CHECKPOINTS_PATH"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image file for evaluation.')
    return parser.parse_args()


class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['NEUTRAL', 'HAPPY', 'SAD', 'SURPRISE', 'FEAR', 'DISGUST', 'ANGER', 'CONTEMPT']
        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        checkpoint = torch.load(checkpoints_path,
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)
        return faces

    def fer(self, frame):
        img0 = Image.fromarray(frame)
        #img0 = Image.open(path).convert('RGB')
        faces = self.detect(img0)
        if len(faces) == 0:
            return 'null'
        ##  single face detection
        x, y, w, h = faces[0]
        # print(img0)
        # print(x,y,w,h)
        #img = img0[y:y+h, x:x+w]
        img = img0.crop((x, y, x+w, y+h))
        img = self.data_transforms(img)
        img = img.view(1, 3, 224, 224)
        img = img.to(self.device)
        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]
            return label


# define a video capture object
if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    model = Model()

    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        label = model.fer(frame)
        print(f'emotion label: {label}')
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()