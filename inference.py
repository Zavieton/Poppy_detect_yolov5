
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

def inference(path, thres=0.25, model_path = './weights/best.pt'):
    device = select_device('')
 
    ret = []  # detection result

    imgsz = 1280
    model = attempt_load(model_path, map_location=device)
    stride = int(model.stride.max()) 
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(path, img_size=imgsz, stride=stride)
    names = ['Fruit stage', 'Seedling stage']
    colors = [(0,0,255), (255,0,0)]

    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, thres, 0.45, classes=None, agnostic=False)
        for pr in pred:
            ret.append(pr.cpu().tolist())

        im0 = im0s
        for _, det in enumerate(pred):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=10)

    
    for r in ret:
        print('\n')
        print('-------------------------')
        for each in r:
            print('Detection Location is (pixel) ')
            print('x1: {}, y1: {}, x2: {}, y2: {} '.format(int(each[0]), int(each[1]), int(each[2]), int(each[3])))
            print('Detection class is {}: {} \n'.format(each[-1], names[int(each[-1])]))

    # Show Image
    im0 = cv2.resize(im0, (800, 600))
    cv2.namedWindow("source", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("source", im0)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = './image/gq_low_003.JPG'
    confidence = 0.25
    model_path = './weights/best.pt'
    inference(path, confidence, model_path)