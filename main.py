import mlp
import cv2
import os
import numpy as np

emotions = ["AN", "DI", "FE", "HA", "SA", "NE", "SU"]

def load_img_by_emotion(dir_name):
        img_list = [f for f in os.listdir(dir_name)]
        img_per_emo = {}
        for emo in emotions:
                img_per_emo[emo] = [img for img in img_list if emo in img]
        return img_per_emo

def pixel_list(img):
        res = []
        for line in img:
                for px in line:
                        res.append(px)
        return res

def img_to_sample(img_filename):
        img = cv2.imread(img_filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        small = cv2.resize(gray_img, (60,50))
        h,w = gray_img.shape
        pixels = pixel_list(gray_img)
        sample = np.zeros(1, dtype = [('input', float, len(pixels)), ('output', float ,1)])
        sample[0] = tuple(pixels), 1
        pack = (img_filename, sample)
        return pack

def main():
        img_dico = load_img_by_emotion("images")
        (name,sample) = img_to_sample("images/KA.HA1.29.tiff")
        size = len(sample[0]['input'])
        network = mlp.MLP(size,256,256,1)
        network.reset()
        network.learn(sample, 1000)
        tests = []
        tests.append(img_to_sample("images/KA.HA2.30.tiff"))
        tests.append(img_to_sample("images/KA.NE1.26.tiff"))
        tests.append(img_to_sample("images/KL.NE1.155.tiff"))
        tests.append(img_to_sample("images/KL.HA1.158.tiff"))
        
        for n,t in tests:
                print "Testing ", n
                network.Test(t)
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
	main()
