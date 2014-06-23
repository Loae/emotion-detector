import mlp
import cv2
import numpy as np


emotions = ["AN", "DI", "FE", "HA", "SA", "NE", "SU"]

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
        return sample

def main():
        sample = img_to_sample("images/KA.HA1.29.tiff")
        size = len(sample[0]['input'])
        network = mlp.MLP(size,256,256,1)
        network.reset()
        network.learn(sample, 1000)
        tests = []
        tests.append(("images/KA.HA2.30.tiff", img_to_sample("images/KA.HA2.30.tiff")))
        tests.append(("images/KA.NE1.26.tiff", img_to_sample("images/KA.NE1.26.tiff")))
        tests.append(("images/KL.NE1.155.tiff", img_to_sample("images/KL.NE1.155.tiff")))
        tests.append(("images/KL.HA1.158.tiff", img_to_sample("images/KL.HA1.158.tiff")))
        
        for n,t in tests:
                print "Testing ", n
                network.Test(t)
        cv2.destroyAllWindows()
        


if __name__ == '__main__':
	main()
