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

    def predict(self, image: Image) -> list(dict):
        features = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**features)
        bboxes = self.visualize_predictions(image, outputs, threshold=self.threshold)
        return bboxes