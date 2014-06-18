import mlp
import cv2
import numpy as np


def pixel_list(img):
        res = []
        for line in img:
                for px in line:
                        res.append(px)
        return res

def img_to_sample(img_filename):
        img = cv2.imread(img_filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray_img, (60,50))
        h,w = small.shape
        pixels = pixel_list(small)
        sample = np.zeros(1, dtype = [('input', float, len(pixels)), ('output', float ,1)])
        sample[0] = tuple(pixels), 1
        return sample

def main():
        sample = img_to_sample("images/smiling-woman.jpg")
        size = len(sample[0]['input'])
        network = mlp.MLP(size,size,1)
        network.learn(sample, 100)
        test1 = img_to_sample("images/smiling-man.jpg")
        network.Test(test1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


if __name__ == '__main__':
	main()
