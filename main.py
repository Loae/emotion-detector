from ep import *
import cv2

def main():
        detector = EmotionPerceptron()
        detector.set_dir("images")
        detector.learn_emo_people("HA",1)
        tests = []
        tests = detector.all_img()
        detector.test_list_img(tests)
        while(True):
                detector.test_from_cam()
                c = cv2.WaitKey(30)
                if c != -1 :
                    break
                
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
	main()
