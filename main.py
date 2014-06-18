import mlp
import cv2
import numpy as np

def pixel_list(img):
        res = []
        for line in img:
                for px in line:
                        res.append(px)
        print len(res)
        return res

def main():
        img = cv2.imread("images/smiling-woman.jpg")
        #        cv2.imshow("test", img)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray_img)
        h,w = gray_img.shape
        print h,w
        pixels = pixel_list(gray_img)
        #print pixels
        #network = mlp.MLP(h*w,h*w,1)
        #samples = np.zeros(1, dtype = [('input', float, 1), ('output', float 1)])
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


if __name__ == '__main__':
	main()
