import cv2
import os
from pdf2image import convert_from_path
import layoutparser as lp
import pandas as pd
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from PIL import Image


output_dir = '../pages_chomage'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = convert_from_path('../chomage.pdf')

for i, image in enumerate(images):
    output_path = os.path.join(output_dir, f'page{i}.jpg')
    if not os.path.exists(output_path):
        image.save(output_path, 'JPEG')


image = cv2.imread("../pages_chomage/page7.jpg")
image = image[..., ::-1]
'''
model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
'''
model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
layout = model.detect(image)
print(layout)


x_1=0
y_1=0
x_2=0
y_2=0

for l in layout:
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

im = cv2.imread('../pages_chomage/page7.jpg')
cv2.imwrite('cropped/chomage.jpg', im[y_1:y_2,x_1:x_2])


# Load the DePlot model and processor
model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")

# Specify the path to the input image
image_path = "cropped/chomage.jpg"

# Load the image from the specified path
image = Image.open(image_path)

# Generate underlying data table from the image

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)

# Decode the predictions to get the extracted data
extracted_data = processor.decode(predictions[0], skip_special_tokens=True)

print(extracted_data)


# Split the extracted data into rows
rows = extracted_data.split("<0x0A>")

# Remove empty strings and strip whitespace from each row
rows = [row.strip() for row in rows if row.strip()]

# Extract column names
column_names = ["Année", "Urbain", "Rural", "Ensemble"]

# Initialize lists for data
years = []
urban = []
rural = []
ensemble = []

# Iterate over rows and extract data
for row in rows[2:]:
    cols = row.split("|")
    years.append(cols[0].strip())
    urban.append(cols[1].strip())
    rural.append(cols[2].strip())
    ensemble.append(cols[3].strip())

# Create DataFrame
df = pd.DataFrame({
    "Année": years,
    "Urbain": urban,
    "Rural": rural,
    "Ensemble": ensemble
})

# Remove leading and trailing whitespace from column names
df.columns = df.columns.str.strip()
output='excel'
if not os.path.exists(output):
    os.makedirs(output)
# Save the DataFrame to an Excel file
excel_file = "excel/chomage.xlsx"
df.to_excel(excel_file, index=False)

print("Excel file saved successfully:", excel_file)




