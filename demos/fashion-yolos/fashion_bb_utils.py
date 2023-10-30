from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
from PIL import Image
MODEL_NAME = "valentinafeve/yolos-fashionpedia"
FEATURE_EXTRACTOR_NAME = "hustvl/yolos-small"

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

def idx_to_text(i):
    return cats[i]

class FashionBoundingBoxPredictor:

    def __init__(self, threshold):
        self.feature_extractor = YolosFeatureExtractor.from_pretrained(FEATURE_EXTRACTOR_NAME)
        self.model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
        self.threshold = threshold

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def visualize_predictions(self, image, outputs, threshold=0.8):
        # keep only predictions with confidence >= threshold
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        # convert predicted boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

        results = []
        for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled.tolist()):
            results.append({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "confidence": p,
                "label": idx_to_text(p.argmax())
            
            })
        return results

    def predict(self, image: Image):
        features = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**features)
        bboxes = self.visualize_predictions(image, outputs, threshold=self.threshold)
        return bboxes
    

"""
write a function that reads the finetuning/export-data-pinterest-600.jsonl file
and converts into a parquet file with the following columns:
{ "bbox_id": [ 150311, 150312, 150313, 150314 ], "category": [ 23, 23, 33, 10 ], "bbox": [ [ 445, 910, 505, 983 ], [ 239, 940, 284, 994 ], [ 298, 282, 386, 352 ], [ 210, 282, 448, 665 ] ], "area": [ 1422, 843, 373, 56375 ] }
the ids can be random generated and the bbox are xmin, ymin, xmax, ymax in pascal voc format.
the original file is a jsonl file with the following format:
{"imageGcsUri":"gs://nh-intern-skroll/update/47287864829918906","boundingBoxAnnotations":[{"displayName":"neckline","xMin":0.052823315118397086,"xMax":0.45901639344262296,"yMin":0.4635627530364373,"yMax":0.7044534412955465,"annotationResourceLabels":{"aiplatform.googleapis.com/annotation_set_name":"8361330731422580736"}}],"dataItemResourceLabels":{}}
where the boundingBoxAnnotations have xmin, ymin, xmax, ymax in normalised manner and the image is in a gcs uri
"""
import json
import pandas as pd
import numpy as np
from google.cloud import storage
import io
import datasets

client = storage.Client()
bucket = client.get_bucket("nh-intern-skroll")
bbbox_id_counter = 1000
def convert_jsonl_to_hfdataset(jsonl_file):
    global bbbox_id_counter
    with open(jsonl_file, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        new_row = {}
        json_line = json.loads(line)
        imageGcsUri = json_line["imageGcsUri"]
        _, blob_name = imageGcsUri.replace("gs://", "").split("/", 1)
        blob = bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes))
        image_w, image_h = image.size
        new_row["image"] = image
        new_row["width"] = image_w
        new_row["height"] = image_h
        new_row["objects"] = {"bbox_id": [], "category": [], "bbox": [], "area": []}
        for bbox in json_line["boundingBoxAnnotations"]:
            bbbox_id_counter += 1
            bbox_id = bbbox_id_counter
            category = cats.index(bbox["displayName"])
            xmin = int(bbox["xMin"] * image_w)
            xmax = int(bbox["xMax"] * image_w)
            ymin = int(bbox["yMin"] * image_h)
            ymax = int(bbox["yMax"] * image_h)
            area = (xmax - xmin) * (ymax - ymin)
            new_row["objects"]["bbox_id"].append(bbox_id)
            new_row["objects"]["category"].append(category)
            new_row["objects"]["bbox"].append([xmin, ymin, xmax, ymax])
            new_row["objects"]["area"].append(area)
        data.append(new_row)
    hf_dataset = datasets.Dataset.from_list(data)
    return hf_dataset, data

# hf_dataset, data = convert_jsonl_to_hfdataset("finetuning/export-data-pinterest-600.jsonl")
# hf_dataset.push_to_hub("meher92/neckline_sleeve_imgs_600", private=True)
