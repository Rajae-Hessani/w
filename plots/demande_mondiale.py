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
second_text_block = layout._blocks[2]

# Afficher les détails du deuxième bloc de texte
print(second_text_block)

# Créer un Layout contenant uniquement le deuxième TextBlock
layout_containing_second_block = lp.Layout([second_text_block])


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
print(x_1,y_1,x_2,y_2)

im = cv2.imread('../pages_rapport_ec/page123.jpg')
cv2.imwrite('cropped/demande_mondiale.jpg', im[y_1:y_2,x_1:x_2])

# Load the DePlot model and processor
model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")

# Specify the path to the input image
image_path = "cropped/demande_mondiale.jpg"

# Load the image from the specified path
image = Image.open(image_path)

# Generate underlying data table from the image

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)

# Decode the predictions to get the extracted data
extracted_data = processor.decode(predictions[0], skip_special_tokens=True)

print(extracted_data)


# Séparer les lignes de données
lines = extracted_data.split('<0x0A>')

# Initialiser une liste pour stocker les données
data = []

# Parcourir les lignes à partir de la deuxième ligne (la première ligne contient les en-têtes)
for line in lines[2:]:
    # Diviser chaque ligne en colonnes en utilisant '|' comme délimiteur
    columns = line.split('|')
    # Extraire l'année et la demande mondiale adressée au Maroc
    year = columns[0].strip()
    demand = columns[1].strip()
    # Ajouter les données à la liste
    data.append([year, demand])

# Créer un DataFrame pandas à partir des données
df = pd.DataFrame(data, columns=['Année', 'Demande mondiale adressée au Maroc'])

output='excel'
if not os.path.exists(output):
    os.makedirs(output)

# Écrire le DataFrame dans un fichier Excel
output_excel_path = 'excel/demande_mondiale_maroc.xlsx'
df.to_excel(output_excel_path, index=False)

print(f"Les données ont été exportées vers le fichier Excel : {output_excel_path}")

