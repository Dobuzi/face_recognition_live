import face_recognition
import cv2
import numpy as np
from os import listdir

class FaceRecognizer(object):
	"""
	This class creates a new frame to recognize the face.
	"""
	def __init__(self, img_path='./face_recognizer/image/'):
		self._frame = None 
		self._frame_to_recognize = None
		self._img_types = ['jpg', 'jpeg', 'png']
		self._img_path = img_path
		self._img_files = []

		self._set_image_files()
		
		self._known_names = []
		self._known_encodings = []
		
		self._set_known_faces()
		
		self._face_locations = [] 
		self._face_encodings = None
		self._face_names = []

	def set_frame(self, frame):
		self._frame = frame
		self._frame_to_recognize = cv2.resize(frame, (0, 0), fx=.25, fy=.25)[:,:,::-1]

	def get_frame(self):
		return self._frame

	def _set_image_files(self):
		self._img_files = [self._img_path + fn for fn in listdir(self._img_path) if fn.split('.')[-1] in self._img_types]

	def _set_known_faces(self):
		for img_file in self._img_files:
			known_name = img_file.split('.')[-2].split('/')[-1]
			image = face_recognition.load_image_file(img_file)
			face_encoding = face_recognition.face_encodings(image)[0]

			print(f'{known_name} is enlisted.')
		
			self._known_names.append(known_name)
			self._known_encodings.append(face_encoding)
	
	def clear(self):
		self._face_locations = []
		self._face_encodings = None
		self._face_names = []

	def recognize(self):
		self._face_locations = face_recognition.face_locations(self._frame_to_recognize)
		self._face_encodings = face_recognition.face_encodings(self._frame_to_recognize, self._face_locations)
		self._face_names = []
		for face_encoding in self._face_encodings:
			matches = face_recognition.compare_faces(self._known_encodings, face_encoding)
			name = "Unknown"
			face_distances = face_recognition.face_distance(self._known_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = self._known_names[best_match_index]
			
			print(f'{name} is on frame')
			self._face_names.append(name)

	def draw(self):
		for name, (top, right, bottom, left) in zip(self._face_names, self._face_locations):
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

			COLOR_BORDER = (0, 0, 255)
			COLOR_TEXT = (255, 255, 255)

			cv2.rectangle(self._frame, (left, top), (right, bottom), COLOR_BORDER, 2)
			cv2.rectangle(self._frame, (left, bottom - 35), (right, bottom), COLOR_BORDER, cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(self._frame, name, (left + 6, bottom - 6), font, 1.0, COLOR_TEXT, 1)
