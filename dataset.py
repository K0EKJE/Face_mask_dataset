import pandas as pd
import numpy as np
import cv2
import glob
from xml.etree import ElementTree

def data_import1(XML_FILES):
  information = {'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'label': [], 'file': [], 'width': [], 'height': []}

  for annotation in XML_FILES:
    tree = ElementTree.parse(annotation)                      
    
    for element in tree.iter():      
      if 'size' in element.tag:    
        for attribute in list(element): 
          if 'width' in attribute.tag: 
            width = int(round(float(attribute.text)))
          if 'height' in attribute.tag:
            height = int(round(float(attribute.text)))    

      if 'object' in element.tag: #object：
        for attribute in list(element):
                
          if 'name' in attribute.tag:
            name = attribute.text                 
            information['label'] += [name]
            information['width'] += [width]
            information['height'] += [height] 
            information['file'] += [annotation.split('/')[-1][0:-4]] 
                            
          if 'bndbox' in attribute.tag:
            for dimension in list(attribute):
              if 'xmin' in dimension.tag:
                xmin = int(round(float(dimension.text)))
                information['xmin'] += [xmin]
              if 'ymin' in dimension.tag:
                ymin = int(round(float(dimension.text)))
                information['ymin'] += [ymin]                                
              if 'xmax' in dimension.tag:
                xmax = int(round(float(dimension.text)))
                information['xmax'] += [xmax]                                
              if 'ymax' in dimension.tag:
                ymax = int(round(float(dimension.text)))
                information['ymax'] += [ymax]
  
  annotations_info_df = pd.DataFrame(information)

  # Add Annotation and Image File Names
  annotations_info_df['annotation_file'] = annotations_info_df['file'] + '.xml'
  annotations_info_df['image_file'] = annotations_info_df['file'] + '.png'
  # Tidy Grammatical Issue
  annotations_info_df.loc[annotations_info_df['label'] == 'mask_weared_incorrect', 'label'] = 'mask_incorrectly_worn'

  return annotations_info_df



def render_image(image_path, annotations_info_df):
    '''
    
    input: input an image, return image with bounding box
    
    '''
    
    
    image = cv2.imread(image_path)  
    img=image_path.split('/')[-1]
    #print(image.shape)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    bound_box=[]
    
    for i in annotations_info_df[annotations_info_df['image_file']==img].index:
      (xmin,ymin,xmax,ymax)=(annotations_info_df.loc[i].xmin,annotations_info_df.loc[i].ymin,
                 annotations_info_df.loc[i].xmax,annotations_info_df.loc[i].ymax)
      bound_box.append((xmin,ymin,xmax,ymax))  
        
        
      if annotations_info_df.loc[i].label=='with_mask':
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0, 200, 0), 2)
        cv2.putText(image, org = (xmin - 8 , ymin - 8), text = "Mask", 
              fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, 
              color = (0, 200, 0))
      elif annotations_info_df.loc[i].label=='mask_incorrectly_worn':
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255, 255, 0), 2)
        cv2.putText(image, org = (xmin - 8, ymin - 3), text = 'Incorrect', 
              fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, 
              color = (255, 255, 0))
      else:
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (200, 0, 0), 2)
        cv2.putText(image, org = (xmin - 8, ymin - 3), text = 'No mask', 
              fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, 
              color = (200, 0, 0))
    
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()
    
    return bound_box,image

def image_crop(annotations_info_df):

  annotations_info_df['cropped_image_file'] = annotations_info_df['file']
  
  for i in range(len(annotations_info_df)):
    # Get The File Path and Read The Image
    image_filepath = os.path.join('/content/Face_mask_dataset/images', annotations_info_df['image_file'].iloc[i])
    image = cv2.imread(image_filepath)
    
    # Set The Cropped Image File Name
    annotations_info_df['cropped_image_file'].iloc[i] = annotations_info_df['cropped_image_file'].iloc[i] + '-' + str(i) + '.png'
    cropped_image_filename = annotations_info_df['cropped_image_file'].iloc[i]
    
    # Get The xmin, ymin, xmax, ymax Value (Bounding Box) to Crop Image
    xmin = annotations_info_df['xmin'].iloc[i]
    ymin = annotations_info_df['ymin'].iloc[i]
    xmax = annotations_info_df['xmax'].iloc[i]
    ymax = annotations_info_df['ymax'].iloc[i]

    # Crop The Image Based on The Values Above
    cropped_image = image[ymin:ymax, xmin:xmax]
    
    cropped_image_directory = os.path.join('/content/Face_mask_dataset/cropped_images', cropped_image_filename) 
    cv2.imwrite(cropped_image_directory, cropped_image)
  
  return annotations_info_df