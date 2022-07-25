from app import app
from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch
import torchvision
import cv2
import numpy as np
import albumentations as albu

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        max_det=300,
                        image_size=640):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        boxes, scores = x[:, :4], x[:, 4]  # boxes, scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output

def plot_preds(numpy_img, preds, border=2):
    img = numpy_img.copy()
    height, width = img.shape[:2]
    for bbox in preds:
        box = np.copy(bbox)
        box = box.astype(int)
        img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),border)
    return img

def find_faces(model, img, img_path, image_size=640, conf_thresh=0.5, transforms=None):
    model.eval()
    height, width = img.shape[:2]
    ratio = max(height, width) / image_size
    image = np.copy(img)
    if transforms is not None:
        sample = transforms(image=image)
        image = sample['image']
    pred = model(image[None,:])
    pred = non_max_suppression(pred[0])
    pred = pred[0]
    boxes = pred.cpu().detach().numpy()
    boxes = boxes[:,:4][boxes[:,4] > conf_thresh]
    boxes *= ratio
    if height < width:
        boxes[:,1] = boxes[:,1] - (width-height)/2
        boxes[:,3] = boxes[:,3] - (width-height)/2
    else:
        boxes[:,0] = boxes[:,0] - (height-width)/2
        boxes[:,2] = boxes[:,2] - (height-width)/2
    img_with_boxes = plot_preds(img, boxes)
    cv2.imwrite(img_path, img_with_boxes)
    del pred, image, img_with_boxes

IMAGE_SIZE = 640
class ToTensorNormalTest(object):
    def __call__(self, image, force_apply=None):
        image = image.transpose((2, 0, 1))
        return {'image': torch.tensor(image, dtype=torch.float32)/255.}

test_transform = albu.Compose([
    albu.LongestMaxSize(max_size=IMAGE_SIZE),
    albu.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
    ToTensorNormalTest()
    ]
)

checkpoint_path = 'app/NNs/checkpoint_yolov5_last.pth'

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True, classes=1, autoshape=False)
model = torch.load(checkpoint_path, map_location=torch.device('cpu'))

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = 'app/static/images/'
app.config['SECRET_KEY'] = 'secret_key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

example_image = 'images/example.bmp'

@app.route('/')
def index():
    image = request.args.get('image')
    if not image:
        image = example_image
    return render_template('index.html', image=image)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = app.config['UPLOAD_FOLDER'] + filename
        file.save(img_path)
        img = cv2.imread(img_path)
        img = find_faces(model, img, img_path, conf_thresh=0.5, transforms=test_transform);
        return redirect(url_for('index', image='images/'+filename))
    return redirect(url_for('index'))