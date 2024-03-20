import cv2
import os
from pdf2image import convert_from_path
import layoutparser as lp
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import pandas as pd

output_dir = '../pages_echange_ext'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = convert_from_path('../echange_ext.pdf')

for i, image in enumerate(images):
    output_path = os.path.join(output_dir, f'page{i}.jpg')
    if not os.path.exists(output_path):
        image.save(output_path, 'JPEG')


image = cv2.imread("../pages_echange_ext/page1.jpg")
image = image[..., ::-1]

model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

layout = model.detect(image)
print(layout)

second_text_block = layout._blocks[7]
print(second_text_block)

layout_containing_second_block = lp.Layout([second_text_block])

second_text_block.block.x_2 = 700
second_text_block.block.y_2 = 900
second_text_block.block.x_1 = 50
second_text_block.block.y_1 = 350


x_1=0
y_1=0
x_2=0
y_2=0

for l in layout_containing_second_block:
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

im = cv2.imread('../pages_echange_ext/page1.jpg')
cv2.imwrite('cropped/balance_commerciale.jpg', im[y_1:y_2,x_1:x_2])

model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")

image_path = "cropped/balance_commerciale.jpg"

image = Image.open(image_path)


inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)

extracted_data = processor.decode(predictions[0], skip_special_tokens=True)

print(extracted_data)

lines = extracted_data.split("<0x0A>")

table = [line.split(" | ") for line in lines]

table = [[item for item in row if item] for row in table]

df = pd.DataFrame(table[1:])

output='excel'
if not os.path.exists(output):
    os.makedirs(output)

excel_file = "excel/import_export.xlsx"
df.to_excel(excel_file, index=False)

