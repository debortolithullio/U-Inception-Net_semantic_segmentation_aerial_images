import numpy as np
import cv2
from tqdm import tqdm

def coordinate2pixel(coordinate, image_jpw):
    '''
      Function to transform  pair of coordinates into pixel values, given the World File (JPW).
      :param coordinate: (x1, y1) coordinates
      :param image_jpw: World file
          source: http://webhelp.esri.com/arcims/9.3/General/topics/author_world_files.htm
          Array image_jpw: [A, D, B, E, C, F]
                source : http://webhelp.esri.com/arcims/9.3/General/topics/author_world_files.htm
                The values are inputs into the following formulas:
  
                x1 = Ax + By + C
                y1 = Dx + Ey + F
  
                where the variables represent the following:
  
                x1 = calculated x-coordinate of the pixel on the map
                y1 = calculated y-coordinate of the pixel on the map
                x = column number of a pixel in the image
                y = row number of a pixel in the image
                A = x-scale; dimension of a pixel in map units in x direction
                B,D = rotation terms
                C,F = translation terms; x,y map coordinates of the center of the upper-left pixel
                E = negative of y-scale; dimension of a pixel in map units in y direction
                
                |x| = |A B|^-1 @ |x1-C| 
                |y|   |D E|      |y1-F|
                
                pixel = scale_matrix^-1 * translation_matrix
    '''
    
    scale_matrix = np.array([[image_jpw[0], image_jpw[2]], [image_jpw[1], image_jpw[3]]])
    translation_matrix = np.array([[coordinate[0]-image_jpw[4]], [coordinate[1]-image_jpw[5]]])
    
    pixel = np.linalg.inv(scale_matrix) @ translation_matrix
    
    return np.around(pixel)
  
def pixel2coordinate(pixel, image_jpw):
    '''
      Function to transform  pair of pixels into coordinate values, given the World File (JPW).
      :param pixel: (x, y) pixel to transform
      :param image_jpw: world file. 
          source: http://webhelp.esri.com/arcims/9.3/General/topics/author_world_files.htm
          Array image_jpw: [A, D, B, E, C, F]
                source : http://webhelp.esri.com/arcims/9.3/General/topics/author_world_files.htm
                The values are inputs into the following formulas:
  
                x1 = Ax + By + C
                y1 = Dx + Ey + F
  
                where the variables represent the following:
  
                x1 = calculated x-coordinate of the pixel on the map
                y1 = calculated y-coordinate of the pixel on the map
                x = column number of a pixel in the image
                y = row number of a pixel in the image
                A = x-scale; dimension of a pixel in map units in x direction
                B,D = rotation terms
                C,F = translation terms; x,y map coordinates of the center of the upper-left pixel
                E = negative of y-scale; dimension of a pixel in map units in y direction
    '''
    
    x1 = (image_jpw[0]*pixel[0] + image_jpw[2]*pixel[1] + image_jpw[4])
    y1 = (image_jpw[1]*pixel[0] + image_jpw[3]*pixel[1] + image_jpw[5])
    
    coordinate = (x1, y1)
    
    return coordinate
  
def check_coord_in_image(img_min_coords, img_max_coords, coordinate):
    if coordinate[0] >= img_min_coords[0] and coordinate[0] <= img_max_coords[0] and coordinate[1] >= img_min_coords[1] and coordinate[1] <= img_max_coords[1]:
      return True
    else:
      return False

def divide_into_chunks(img, desired_size):
      '''
      Function to get chunks from image
      :param img: Image to be processed
      :param desired_size: Size of the output images
      :return: Array with img.size[0]/desired_size images
      '''

      if img.shape[0]%desired_size != 0:
            new_shape = (img.shape[0]//desired_size+1)*desired_size
            img = cv2.resize(img, (new_shape, new_shape), interpolation=cv2.INTER_AREA)

      chunks = []  

      for i in range(0, img.shape[0], desired_size):
            for j in range(0, img.shape[1], desired_size):
                  chunks.append(img[i:i+desired_size, j:j+desired_size])

      return chunks

def dice_coef_float(y_true, y_pred):
  y_true = y_true.ravel()
  y_pred = y_pred.ravel()
  
  intersection = np.sum(y_true * y_pred)
  
  return (2.0 * intersection + 1.0) / (np.sum(y_true) + np.sum(y_pred) + 1.0)