import mlp
import cv2
import os
import numpy as np

emo_text = {"AN": "ANGRY",
                "DI" : "DISGUTS",
                "FE" : "FEAR",
                "HA": "HAPPY",
                "SA": "SAD",
                "NE" : "NEUTRAL",
                "SU": "SURPRISE"}
emo_list = ["ANGRY",
            "DISGUTS",
            "FEAR",
            "HAPPY",
            "SAD",
            "NEUTRAL",
            "SURPRISE"]
class EmotionPerceptron:
    def __init__(self):
        self.emotions = ["AN", "DI", "FE", "HA", "SA", "NE", "SU"]
        self.network = mlp.MLP(256*256,256,256,1)
        self.img_dico = {}
        self.nb_try = 2500
        self.folder = ""
        self.init_mlp(256, 256)
        self.init_cam()
    
    def init_cam(self):
        self.capture = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.capture.read()
        small = cv2.resize(frame, (256, 256))
        return small

        #we add 2 hidden layers
    def init_mlp(self, img_h, img_w):
        self.input_size = img_h*img_w
        self.network = mlp.MLP(self.input_size, img_w, img_h, len(self.emotions))
            
    def load_img_by_emotion(self, dir_name):
        self.folder = dir_name
        img_list = [f for f in os.listdir(dir_name)]
        img_per_emo = {}
        for emo in self.emotions:
            img_per_emo[emo] = [img for img in img_list if emo in img]
        return img_per_emo
        
    def set_dir(self, dir_name):
        self.img_dico = self.load_img_by_emotion(dir_name)

    def pixel_list(self, img):
        res = []
        for line in img:
            for px in line:
                res.append(float(px))
        return res

    def img_to_sample(self, img_filename, label = 0):
        path = "{}/{}".format(self.folder, img_filename)
        print "Loading ", path
        img = cv2.imread(path)
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h,w = gray_img.shape
            pixels = self.pixel_list(gray_img)
            sample = np.zeros(1, dtype = [('input', float, len(pixels)), ('output', float ,len(self.emotions))])
            output = [0,]*len(self.emotions)
            output[label] = 1
            sample['input'] = pixels
            sample['output'] = output
            pack = (img_filename, sample)
            return pack
        else:
            return None

    def frame_to_sample(self, img):
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h,w = gray_img.shape
            pixels = self.pixel_list(gray_img)
            sample = np.zeros(1, dtype = [('input', float, len(pixels)), ('output', float ,len(self.emotions))])
            output = [0,]*len(self.emotions)
            sample['input'] = pixels
            sample['output'] = output
            pack = ("Capture", sample)
            return pack
        else:
            return None

    def img_list_to_samples(self, img_list, label = 0):
         samples = np.zeros(len(img_list), dtype = [('input', float, self.input_size), ('output', float ,len(self.emotions))])
         n = len(img_list)
         for i in xrange(n):
             path = "{}/{}".format(self.folder, img_list[i])
             print "Loading ", path
             img = cv2.imread(path)
             if img is not None:
                 gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                 h,w = gray_img.shape
                 pixels = self.pixel_list(gray_img)
                 output = [0,]*len(self.emotions)
                 output[label] = 1
                 samples['input'][i] = pixels
                 samples['output'][i] = output
             else:
                 return None
         return samples

    def img_list_emo_people(self, emotion, nb_people = -1):
        if (nb_people == -1):
            return self.img_dico[emotion]
        person_list = []
        img_list = []
        for img in self.img_dico[emotion]:
            infos = img.split(".")
            p = infos[0]
            if p not in person_list:
                person_list.append(p)
        for i in xrange(nb_people):
                p = person_list[i]
                for img in self.img_dico[emotion]:
                    if img.startswith(p):
                        img_list.append(img)
        return img_list

    def learn_emo_people(self, emotion, nb_people = -1):
        self.network.reset()        
        img_list = self.img_list_emo_people(emotion, nb_people)
        train_set = self.img_list_to_samples(img_list, self.emotions.index(emotion))
        print "Learning ", img_list
        self.network.learn(train_set, self.nb_try)
    
    def all_img(self):
        res = []
        for img_list in self.img_dico.values():
            for img in img_list:
                res.append(img)
        return res

    def test_list_img(self, tests):
        for f in tests:
            (n, s) = self.img_to_sample(f)
            print "Testing ", n, "size ", s["input"].size
            c = self.network.Test(s)
            print "Found ", self.emotions[c]

    def test_from_cam(self):
        frame = self.get_frame()
        self.test_frame(frame)

    def test_frame(self, frame):
        n, sample = self.frame_to_sample(frame)
        print "Testing From Webcam"
        emo = self.network.Test(sample)
        if emo is not None:
            text = emo_list[emo]
            print "Found ", self.emotions[emo]
            cv2.putText(frame,text, (5,200), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)
            cv2.imshow("Emotion Detector", frame)

        
    def save_to_file(self, filename):
        self.network.save_to_file(filename)


def test():
    detector = EmotionPerceptron()
    detector.set_dir("images")
    while(True):
        detector.test_from_cam()
        c = cv2.waitKey(30)
        if c != -1 :
            break
                
    cv2.destroyAllWindows()
