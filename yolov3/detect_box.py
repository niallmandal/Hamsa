import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import pyautogui
import time

import hand_altering
import sys
pyautogui.FAILSAFE = False

def derivative(previous,current,fps):
    x1,y1 = previous
    x2,y2 = current
    dx = fps*(x2 - x1)
    dy = fps*(y2 - y1)

    return (dx,dy)

def detect(save_txt=False, save_img=False):
    webcam_height = 480
    webcam_top_margin = 60
    webcam_bottom_margin = 420
    frame_height = webcam_bottom_margin-webcam_top_margin
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    data_coords_open = []
    data_coords_closed = []
    opened_distance = []
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s
            original_image = im0.copy()
            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                hand_detect_open = []
                hand_detect_closed = []
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)

                        #custom box code
                        box_np = []
                        for tensor in xyxy:
                            box_np.append(int(tensor))

                        avg_x = int((box_np[0]+box_np[2])/2)
                        avg_y = int((box_np[1]+box_np[3])/2)
                        cent = (avg_x,avg_y)

                        height, width, channels = im0.shape

                        x_len = box_np[2]-box_np[0]
                        x_ratio_from_origin = (box_np[0]-0)/(width-x_len)
                        x_padding_left = x_ratio_from_origin * x_len
                        center_x = int(box_np[0]-0 + x_padding_left)

                        y_len = box_np[3] - box_np[1]
                        y_ratio_from_origin = (box_np[1]-webcam_top_margin)/(frame_height-y_len)
                        y_padding_top = y_ratio_from_origin * y_len
                        center_y = int(box_np[1]-0 + y_padding_top)

                        mouse_x = 1919 * (center_x/width)
                        mouse_y = 1079 * ((center_y-webcam_top_margin)/frame_height)

                        #pyautogui.moveTo(center_x, mouse_y)

                        #im0 = cv2.circle(im0, (center_x,center_y), 3, (255,0,0), 1)

                        hand = {
                            "avg_x": avg_x,
                            "avg_y": avg_y,
                            "center_x": center_x,
                            "center_y": center_y,
                            "mouse_x": mouse_x,
                            "mouse_y": mouse_y,
                            "label": label.split()[0],
                            "accuracy": label.split()[1]
                        }
                        if label.split()[0] == "open":
                            hand_detect_open.append(hand)
                            im0 = cv2.circle(im0, (center_x,center_y), 7, (255,0,0), 3)
                        if label.split()[0] == "closed":
                            hand_detect_closed.append(hand)
                            im0 = cv2.circle(im0, (center_x,center_y), 7, (0,0,255), 3)
                        original_image = im0.copy()
                        #custom code end
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                if len(hand_detect_open) > 0:
                    opened = sorted(hand_detect_open, key=lambda k: k['accuracy'])[0]
                    data_coords_open.append((opened["center_x"],opened["center_y"],opened["mouse_x"],opened["mouse_y"]))
                else:
                    data_coords_open = []
                    opened_distance = []
                if len(hand_detect_closed) > 0:
                    closed = sorted(hand_detect_closed, key=lambda k: k['accuracy'])[0]
                    data_coords_closed.append((closed["center_x"], closed["center_y"],closed["mouse_x"],closed["mouse_y"]))
                else:
                    data_coords_closed = []

                elapsed = time.time() - t
                fps = 1/elapsed
                fps_text = "FPS: " + str(1/elapsed)[:4]

                if len(hand_detect_open) > 0 and len(hand_detect_closed) > 0:
                    im0 = hand_altering.rotate((1920/2,1080/2),data_coords_open, data_coords_closed, opened, closed, im0)
                elif len(hand_detect_open) == 1:
                    #correct middle click
                    pyautogui.mouseUp(button='middle')
                    hand_altering.mouse_move((opened["mouse_x"],opened['mouse_y']))
                elif len(hand_detect_open) > 1:
                    hand_a = opened
                    hand_b = sorted(hand_detect_open, key=lambda k: k['accuracy'])[1]
                    im0, updated_opened_distance, dydx_distance = hand_altering.zoom(hand_a,hand_b,opened_distance,fps,im0)
                    opened_distance = updated_opened_distance
                    pyautogui.scroll(int(dydx_distance))
                elif len(data_coords_closed) > 0:
                    hand_altering.pan((closed["mouse_x"],closed['mouse_y']))
                im0 = cv2.putText(im0, fps_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX , 1,(255, 0, 0) , 2, cv2.LINE_AA)
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow('image with all the data', im0)
                cv2.imshow('flipped imaged', cv2.flip(original_image, 1 ))
            #Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
