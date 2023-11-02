from fastapi import FastAPI, File, Response
from contextlib import asynccontextmanager
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# load os environment variables
MODEL_NAME = os.getenv("MODEL_NAME")
FEATURE_EXTRACTOR_NAME = os.getenv("FEATURE_EXTRACTOR_NAME")

print("MODEL_NAME:", MODEL_NAME)
print("FEATURE_EXTRACTOR_NAME:", FEATURE_EXTRACTOR_NAME)    


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
                "confidence": p.max().item(),
                "label": idx_to_text(p.argmax())
            
            })
        return results

    def draw_bounding_boxes(self, image: Image, bboxes):
        plt.figure(figsize=(16,10))
        plt.imshow(image)
        ax = plt.gca()
        colors = COLORS * 100
        for bb_obj, c in zip(bboxes, colors):
            xmin = bb_obj["xmin"]
            ymin = bb_obj["ymin"]
            xmax = bb_obj["xmax"]
            ymax = bb_obj["ymax"]
            label = bb_obj["label"]
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            ax.text(xmin, ymin, label, fontsize=10,
                    bbox=dict(facecolor=c, alpha=0.8))
        plt.axis('off')
        output_bytes = BytesIO()
        plt.savefig(output_bytes, format="jpeg")
        return output_bytes

    def predict(self, image: Image):
        features = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**features)
        bboxes = self.visualize_predictions(image, outputs, threshold=self.threshold)
        return bboxes


def get_fashion_bb_predictor(threshold=0.5):
    return FashionBoundingBoxPredictor(threshold=threshold)

models = {}

# creating an async context manager to define events to take place on startup and shutdown of the FastAPI app
# more information about lifespan events can be found here: https://fastapi.tiangolo.com/advanced/events/
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup
    models['fashion_bb'] = get_fashion_bb_predictor(threshold=0.8)
    yield
    # Clean up the ML models and release the resources on shutdown
    models.clear()

app = FastAPI(lifespan=lifespan)

# startup checks detect if a container has started successfully which will then kickoff the liveness and readiness checks
@app.get("/startup/", status_code=200)
def startup_check():
    return "Startup check succeeded."

# liveness checks detect deployment containers that transition to an unhealthy state and remedy said situations through targeted restarts
@app.get("/liveness/", status_code=200)
def liveness_check():
    return "Liveness check succeeded."

# readiness checks tell our load balancers when a container is ready to receive traffic
@app.get("/readiness/", status_code=200)
def readiness_check():
    return "Readiness check succeeded."

@app.get("/")
async def root():
    return {"message": "Welcome to TUNED Fashion Bounding Box HF REST API!"}


# Add post endpoint that takes a file as input and returns a json with bounding boxes and labels
@app.post("/predict-bounding-boxes")
async def predict_bounding_boxes(file: bytes = File(...)):
    # read the image file
    image = Image.open(BytesIO(file))
    print("hit /predictboundingboxes with imgsize:", image.size)

    # get the bounding boxes
    bboxes = models['fashion_bb'].predict(image)
    # return the bounding boxes
    return bboxes


# Add post endpoint that takes a file as input and returns a file with bounding boxes and labels
@app.post("/predict-draw-bounding-boxes")
async def predict_and_draw_bounding_boxes(file: bytes = File(...)):
    # read the image file
    image = Image.open(BytesIO(file))
    print("hit /predictanddrawbb with imgsize:", image.size)
    # get the bounding boxes
    bboxes = models['fashion_bb'].predict(image)
    
    # draw the bounding boxes
    output_bytes = models['fashion_bb'].draw_bounding_boxes(image, bboxes)
    return Response(content=output_bytes.getvalue(), media_type="image/jpeg")