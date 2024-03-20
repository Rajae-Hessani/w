import cv2
import os
from pdf2image import convert_from_path
import layoutparser as lp
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import pandas as pd


output_dir = '../pages_rapport_ec'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = convert_from_path('../Rapport_economique_financier.pdf')

for i, image in enumerate(images):
    output_path = os.path.join(output_dir, f'page{i}.jpg')
    if not os.path.exists(output_path):
        image.save(output_path, 'JPEG')


image = cv2.imread("../pages_rapport_ec/page123.jpg")
image = image[..., ::-1]

model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
'''
model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
'''
layout = model.detect(image)
print(layout)

# Accéder au deuxième bloc de texte dans le Layout
first_text_block = layout._blocks[1]

# Afficher les détails du deuxième bloc de texte
print(first_text_block)

# Créer un Layout contenant uniquement le deuxième TextBlock
layout_containing_first_block = lp.Layout([first_text_block])

x_1=0
y_1=0
x_2=0
y_2=0

for l in layout_containing_first_block:
  #print(l)
  if l.type == 'Figure':
    x_1 = int(l.block.x_1)
    print(l.block.x_1)
    y_1 = int(l.block.y_1)
    x_2 = int(l.block.x_2)
    y_2 = int(l.block.y_2)

    break

print(x_1,y_1,x_2,y_2)

output_dir = 'cropped'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

im = cv2.imread('../pages_rapport_ec/page123.jpg')
cv2.imwrite('cropped/cours_brent.jpg', im[y_1:y_2,x_1:x_2])




model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")

image_path = "cropped/cours_brent.jpg"

image = Image.open(image_path)

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)

extracted_data = processor.decode(predictions[0], skip_special_tokens=True)

print(extracted_data)


output='excel'
if not os.path.exists(output):
    os.makedirs(output)



