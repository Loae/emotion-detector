from ep import *
                     
def main():
        detector = EmotionPerceptron()
        detector.set_dir("images")
        detector.learn_emo_people("HA")
        tests = []
        tests = detector.img_list_emo_people("HA")
        detector.test_list_img(tests)
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
	main()
