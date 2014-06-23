from ep import *
                     
def main():
        detector = EmotionPerceptron()
        detector.set_dir("images")

        detector.learn_emo_people("HA",1)
        tests = []
        tests = detector.img_list_emo_people("NE")
        detector.test_list_img(tests)
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
	main()
