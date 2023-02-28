import os
from PIL import Image
out_dir= "pdf/"
source_dir="source/"
for file in os.listdir(source_dir):
    if file .split('.')[-1]in('png','jpg','jpeg'):
        image=Image.open(os.path.join(source_dir,file))
        image_converted = image.convert('RGB')
        image_converted.save(os.path.join(out_dir, '{0}.pdf'.format(file.split('.')[-2])))

