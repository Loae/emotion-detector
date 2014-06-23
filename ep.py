import mlp
import cv2
import os
import numpy as np


class EmotionPerceptron:
    def __init__(self):
        self.emotions = ["AN", "DI", "FE", "HA", "SA", "NE", "SU"]
        self.network = mlp.MLP(256*256,256,256,1)
        self.img_dico = {}
        self.nb_try = 100
        self.folder = ""
        self.init_mlp(256, 256)

        #we add 2 hidden layers
    def init_mlp(self, img_h, img_w):
        self.input_size = img_h*img_w
        self.network = mlp.MLP(self.input_size, img_w, img_h, 1)
            
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

    def img_to_sample(self, img_filename):
        path = "{}/{}".format(self.folder, img_filename)
        print "Loading ", path
        img = cv2.imread(path)
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h,w = gray_img.shape
            pixels = self.pixel_list(gray_img)
            sample = np.zeros(1, dtype = [('input', float, len(pixels)), ('output', float ,1)])
            sample[0] = tuple(pixels), 0
            pack = (img_filename, sample)
            return pack
        else:
            return None

    def img_list_to_samples(self, img_list):
         samples = np.zeros(len(img_list), dtype = [('input', float, self.input_size), ('output', float ,1)])
         n = len(img_list)
         for i in xrange(n):
             path = "{}/{}".format(self.folder, img_list[i])
             print "Loading ", path
             img = cv2.imread(path)
             if img is not None:
                 gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                 h,w = gray_img.shape
                 pixels = self.pixel_list(gray_img)
                 samples[i] = tuple(pixels), float(2*i-n)/(2*n)
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
        train_set = self.img_list_to_samples(img_list)
        print "Learning ", img_list
        self.network.learn(train_set, self.nb_try)
    
    def test_list_img(self, tests):
        for f in tests:
            (n, s) = self.img_to_sample(f)
            print "Testing ", n, "size ", s["input"].size
            self.network.Test(s)
        
