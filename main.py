from ep import *
import cv2

def main():
        detector = EmotionPerceptron()
        detector.set_dir("images")
        detector.load_from_file("SAVE_weights")
#       detector.learn_all_emotions(1)
#       detector.save_to_file("emotion_weights")
        tests = []
        tests = detector.all_img()
        detector.run_scenarios(tests)
        while(True):
                detector.test_from_cam()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
	main()
