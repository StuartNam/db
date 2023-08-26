import os
import shutil

import tqdm
import numpy as np

from deepface import DeepFace

class Accumulator:
    # Accumulator helps with accumulating scalar values. Useful for calculating scores
    def __init__(self):
        self.total = 0
        self.n = 0

    def accumulate(self, value):
        self.total += value
        self.n += 1
    
    def average(self):
        if self.n == 0:
            return 0
        
        result = self.total / self.n

        return result
    
class Metric:
    def __init__(self):
        pass

class FDFR(Metric):
    @classmethod
    def eval(cls, target_folder_path):
        files = os.listdir(target_folder_path)
        num_files = len(files)
        total_faces = num_files

        if num_files == 0:
            return 0
        
        for file in tqdm.tqdm(files, desc = "FDFR.eval()", unit = "image"):
            image_path = os.path.join(target_folder_path, file)
            
            # Find faces in file using RetinaFace
            try:
                DeepFace.extract_faces(
                    img_path = image_path, 
                    target_size = (224, 224), 
                    detector_backend = 'retinaface'
                )

            # If there are no faces
            except:
                
                total_faces -= 1

        
        fdfr = total_faces / num_files

        return fdfr
        
class ISM(Metric):
    @classmethod
    def eval(cls, target_folder_path, identity_folder_path):
        target_files = os.listdir(target_folder_path)

        target_face_vectors = []
        for file in target_files:
            image_path = os.path.join(target_folder_path, file)

            try:
                face_embedding_info = DeepFace.represent(
                    img_path = image_path,
                    model_name = 'ArcFace'
                )   
            except:
                shutil.move(image_path, './data/invalid/')
            else:
                face_vector = face_embedding_info[0]['embedding']
                target_face_vectors.append(face_vector)
        
        identity_face_vectors = []

        identity_files = os.listdir(identity_folder_path)

        for file in identity_files:
            image_path = os.path.join(identity_folder_path, file)

            try:
                face_embedding_info = DeepFace.represent(
                    img_path = image_path,
                    model_name = 'ArcFace'
                )   
            except:
                shutil.move(image_path, './data/invalid/')
            else:
                face_vector = face_embedding_info[0]['embedding']
                identity_face_vectors.append(face_vector)

        def cosine_similarity(vector1, vector2):
            dot_product = np.dot(vector1, vector2)
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            similarity = dot_product / (norm_vector1 * norm_vector2)
            
            return similarity

        big_accumulator = Accumulator()
        for target_face_vector in tqdm.tqdm(target_face_vectors, desc = "ISM.eval()", unit = "image"):  
            small_accumulator = Accumulator()

            for identity_face_vector in identity_face_vectors:
                score = cosine_similarity(target_face_vector, identity_face_vector)
                small_accumulator.accumulate(score)

            big_accumulator.accumulate(small_accumulator.average())

        ism = big_accumulator.average()

        return ism
        


        
