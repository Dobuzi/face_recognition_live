from face_recognizer import FaceRecognizer
import cv2

if __name__=='__main__':
	video_capture = cv2.VideoCapture(0)
	fr = FaceRecognizer()
	process_this_frame = True

	while True:
		ret, frame = video_capture.read()
		fr.set_frame(frame)
		fr.clear()

		if process_this_frame:
			fr.recognize()

		process_this_frame = not process_this_frame

		fr.draw()
		
		cv2.imshow('Video', fr.get_frame())
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()
