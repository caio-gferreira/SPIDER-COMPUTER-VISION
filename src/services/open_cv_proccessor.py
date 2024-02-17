import cv2
import os
import numpy as np
import re

class OpenCVProcessor:
    def get_proccessed_images(self, folder):
        images, labels = self.load_images(folder)

        images_proccessed = self.process_images(images)
        labels_proccessed = self.process_labels(labels)

        return images_proccessed, labels_proccessed

    def load_images(self, folder):
        list_images_spiders = []
        list_labels_spiders_images = []
    
        for filename in os.listdir(folder):
            label_spider_image = filename.split('.')[0]
            image_spider = cv2.imread(os.path.join(folder, filename))  

            if image_spider is not None:
                list_images_spiders.append(image_spider)
                list_labels_spiders_images.append(label_spider_image)
                
        return list_images_spiders, list_labels_spiders_images

    def process_images(self, images_list):
        images_formateds_list = []

        for image in images_list:
           formated_image = self.format_image(image)
           images_formateds_list.append(formated_image)
        
        images_proccesseds_list = np.array(images_formateds_list)

        scaled_image_pixels = images_proccesseds_list / 255.0

        return scaled_image_pixels
    
    def process_labels(self, labels_list):
        labels_proccessed_list = []
        for label in labels_list:

            if re.match('armadeira.*', label):
                labels_proccessed_list.append(0)

            if re.match('viuva.*', label):
                labels_proccessed_list.append(1)

        return np.array(labels_proccessed_list)
    
    def format_image(self, image):
        resized_image = cv2.resize(image, (28, 28))
        recolor_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        formated_image = recolor_image

        return formated_image
