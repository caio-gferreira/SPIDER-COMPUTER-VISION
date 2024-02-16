import cv2
import os
import numpy as np

class ImageProcessor:
    def load_images(self, folder):
        list_images_spiders = []
        list_labels_spiders_images = []
    
        for filename in os.listdir(folder):
            label_spider_image = filename.split('.')[0]
    
            image_spider = cv2.imread(os.path.join(folder, filename))
            image_spider = self.process_images(image_spider)
            
            if image_spider is not None:
                list_images_spiders.append(image_spider)
                list_labels_spiders_images.append(label_spider_image)
    
        return np.array(list_images_spiders), np.array(list_labels_spiders_images)

    def process_images(self, image):
        image_process = cv2.resize(image, (28, 28))
        image_process = cv2.cvtColor(image_process, cv2.COLOR_BGR2GRAY)

        return image_process