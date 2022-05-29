import face_recognition
import cv2
import numpy as np
from os import listdir


def get_image_dirs(image_folder_path='./image/'):
	fns = []
	for fn in listdir(image_folder_path):
		if fn.split('.')[-1] in ['jpg', 'jpeg', 'png']:
			fns.append(image_folder_path+fn)
	return fns

def make_known_face(images_dir):
	names = []
	encodings = []
	
	for img_path in images_dir:
		print(img_path)
		name = img_path.split('.')[-2].split('/')[-1]
		image = face_recognition.load_image_file(img_path)
		face_encoding = face_recognition.face_encodings(image)[0]
		
		names.append(name)
		encodings.append(face_encoding)

	return names, encodings 

def get_face_names(frame, names, encodings):
	face_locations = face_recognition.face_locations(frame)
	face_encodings = face_recognition.face_encodings(frame, face_locations)

	face_names = []
	for face_encoding in face_encodings:
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
		name = "Unknown"

		face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
		print(face_distances)
		best_match_index = np.argmin(face_distances)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]

		face_names.append(name)
	
	return face_names, face_locations

def display_results(frame, locations, names):
	for (top, right, bottom, left), name in zip(locations, names):
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		COLOR_BORDER = (0, 0, 255)
		COLOR_TEXT = (255, 255, 255)

		cv2.rectangle(frame, (left, top), (right, bottom), COLOR_BORDER, 2)
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), COLOR_BORDER, cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, COLOR_TEXT, 1)

	cv2.imshow('Video', frame)

if __name__=='__main__':
	video_capture = cv2.VideoCapture(0)
	images_dir = get_image_dirs()
	known_face_names, known_face_encodings = make_known_face(images_dir)

	face_names = []
	face_locations = []
	process_this_frame = True

	while True:
		ret, frame = video_capture.read()
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		rgb_small_frame = small_frame[:, :, ::-1]

		if process_this_frame:
			face_names, face_locations = get_face_names(rgb_small_frame, known_face_names, known_face_encodings)

		process_this_frame = not process_this_frame
		
		display_results(frame, face_locations, face_names)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()
