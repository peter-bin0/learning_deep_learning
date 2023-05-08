import pyrealsense2 as rs
import numpy as np
import os, datetime
import pandas as pd
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


##############################################################################
#######################################     识别并保存检测图片和果实csv      #######################################
###################################################################################################################
def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))


    # **********新增***********
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir   将标签存储在labels文件夹中
    (save_dir / 'cropped' if out else save_dir).mkdir(parents=True, exist_ok=True)  # 将获取的目标框截图存储在cropped文件夹中
    # **********新增***********

    # Initialize
    set_logging()  # **********新增***********
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # **********新增***********
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                # p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count  # ***********新增***********
            else:
                # p, s, im0 = path, '', im0s
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)  # ***********新增***********

            # ***********新增***********
            im1 = im0.copy()
            p = Path(p)  # to Path
            # save_path = str(Path(out) / Path(p).name)
            save_path = str(save_dir / p.name)  # img.jpg  # ***********新增***********
            # ***********新增***********
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                print("正确的坐标信息")
                print(type(det))
                det = det.cpu()

                det_pix = np.array(det)
                print(det_pix)
                num = det_pix.shape[0]
                print(num)

                data = pd.DataFrame()
                # image_path = 'E:\yolov5\yolov5-master1\deal_img/images/' + str(tname) + '.png'
                image_path = '299740.png'
                for i in range(num):
                    newdata = pd.DataFrame(0, index=range(1),
                                           columns=['filename',
                                                    'xmin', 'ymin',
                                                    'xmax', 'ymax',
                                                    'xcenter', 'ycenter',
                                                    'scores',
                                                    'class'
                                                    ])
                    newdata.iloc[0, 0] = image_path.split("\\")[-1].split('.')[0]
                    newdata.iloc[0, 1] = int(det_pix[i][0])  # xmin
                    newdata.iloc[0, 2] = int(det_pix[i][1])  # ymin
                    newdata.iloc[0, 3] = int(det_pix[i][2])  # xmax
                    newdata.iloc[0, 4] = int(det_pix[i][3])  # ymax
                    newdata.iloc[0, 5] = int(((det_pix[i][0]) + (det_pix[i][2])) / 2)  # xcenter
                    newdata.iloc[0, 6] = int((det_pix[i][1] + det_pix[i][3]) / 2)  # ycenter
                    newdata.iloc[0, 7] = float(det_pix[i][4])
                    newdata.iloc[0, 8] = int(det_pix[i][5])
                    print("*****************************", newdata.iloc[0, 7])
                    if ((newdata.iloc[0, 8] == 0) or (newdata.iloc[0, 8] == 1)):
                         data = data.append(newdata)
                data.to_csv('E:\yolov5\yolov5-5.0-master1\deal_img\out_put_csv/out_csv.csv', index=False)
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                # Write results
                k = 0
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    if out:
                        x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                        img_ = im1.astype(np.uint8)
                        crop_img = img_[y:y + h, x:x + w]

                        # !!rescale image !!!
                        filename = p.name
                        filename_no_extesion = filename.split('.')[0]
                        extension = filename.split('.')[1]
                        new_filename = str(filename_no_extesion) + '_' + str(k) + '.' + str(extension)
                        dir_path = os.path.join(save_dir, 'cropped')
                        filepath = os.path.join(dir_path, new_filename)
                        print(filepath)
                        cv2.imwrite(filepath, crop_img)
                        k = k + 1
                    else:
                        print("There is no detected object")
                        continue

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

if __name__ == '__main__':

    time_start = time.time()
    tname = str(time_start)[5:10]
    print(tname)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    # 获取图像，realsense刚启动的时候图像会有一些失真，我们保存第100帧图片。
    for i in range(30):
        data = pipeline.wait_for_frames()
        data = align.process(data)
        depth = data.get_depth_frame()
        color = data.get_color_frame()

    # 获取内参
    dprofile = depth.get_profile()
    cprofile = color.get_profile()

    cvsprofile = rs.video_stream_profile(cprofile)
    dvsprofile = rs.video_stream_profile(dprofile)

    color_intrin = cvsprofile.get_intrinsics()
    print("*****************************************************")
    print(color_intrin)
    depth_intrin = dvsprofile.get_intrinsics()

    print(depth_intrin)
    depth_image = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color.get_data())

    cv2.imwrite('/Users/binpan/code/yolov5-master1/deal_img/images/' + str(tname) + '.png', color_image)
    cv2.imwrite('/Users/binpan/code/yolov5-master1/deal_img/depth/' + str(tname) + '.png', depth_image)
    end1 = time.time()
    print("采集图片用时", end1 - time_start)

    img = cv2.imread('/Users/binpan/code/yolov5-master1/deal_img/images/' + str(tname) + '.png', -1)
    # print(img.shape)  # Print image shape
    # cv2.imshow("original", img)

    img1 = cv2.imread('/Users/binpan/code/yolov5-master1/deal_img/depth/' + str(tname) + '.png', -1)
    print("**************读取深度图像中每个像素点的深度信息**************")
    print(img1)

    xz = img.shape
    print(xz[0])   #480
    print(xz[1])   #640


    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='E:\yolov5\yolov5-5.0-master1/runs/train\ymj_exp28\weights/best.pt', help='model.pt path')     # 识别番茄果实的模型
    parser.add_argument('--source', type=str, default='E:\yolov5\yolov5-5.0-master1\deal_img/images/' + str(tname) + '.png',help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='E:\yolov5\yolov5-5.0-master1\deal_img/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--out', action='store_false', help='save the detected object as separate image')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        end3 = time.time()
        detect()
    end2 = time.time()
    print("检测用时", end2 - end3)
    if os.path.exists('/Users/binpan/code/yolov5-master1/deal_img/out_put_csv/out_csv.csv'):
        if os.path.getsize('/Users/binpan/code/yolov5-master1/deal_img/out_put_csv/out_csv.csv'):
            print('文件存在且不为空')

            # 读取框的csv
            data = pd.read_csv('/Users/binpan/code/yolov5-master1/deal_img/out_put_csv/out_csv.csv')
            # 将读取到的数据转化为数组
            a = data.values.reshape(-1, 9)
            row = a.shape[0]
            print(row)
            print("**************读取存储果实像素信息的csv表格*******************")
            print(a)
            xmin = int(a[0][1])
            ymin = int(a[0][2])
            xmax = int(a[0][3])
            ymax = int(a[0][4])
            xc = int(a[0][5])
            yc = int(a[0][6])
            x1 = int(a[0][5] - 5)
            x2 = int(a[0][5] + 5)
            y1 = int(a[0][6] - 5)
            y2 = int(a[0][6] + 5)
            w = int(xmax - xmin)
            h = int(ymax - ymin)

            print(xmin,xmax,ymin,ymax)

            # p = np.random.randint(0, 1, w*h)
            p = []
            p2 = []
            ki = 0
            sum = 0
            sum1 = 0
            sum2 = 0
            kj = 0
            ka = 0
            kb = 0
            kc = 0
            kd = 0
            ke = 0
            kf = 0

            for ix in range(xmin, xmax+1):
                for iy in range(ymin, ymax+1):
                    h_i = float(img1[iy][ix])
                    p.append(h_i)
            p1 = sorted(p)

            # print("深度统计", p)
            # print("深度统计", p1)

            for xi in range(x1, x2):
                for yi in range(y1, y2):
                    h_i1 = float(img1[yi][xi])
                    if(h_i1>50):
                        sum = sum + h_i1
                        ki = ki + 1

            print(ki)
            rav = int(sum / ki);
            print("中心平均深度",rav)
            # rav = float(img1[yc][xc])
            # print("中心深度", rav)

            # img = cv2.medianBlur(img, 5)  # 中值滤波去除噪点
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 原图 从彩色图变单通道灰度图像
            cv2.imshow("gray", gray)

            # t,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)   #灰度图像 转 二值图像 ****t, 多变量赋值*****可改进：Ostu
            # t, binary = cv2.threshold(gray, 0, 255,
            #                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 灰度图像 转 二值图像 ****t, 多变量赋值*****可改进：Ostu
            # cv2.imshow("binary", binary)  # 显示二值化图像

            for ix1 in range(0, xz[0]):
                for iy1 in range(0, xz[1]):
                    if((iy1>=xmin and iy1<=xmax) and (ix1>=ymin and ix1<=ymax)):
                        gray[ix1, iy1] = 255
                        ka = ka + 1
                    else:
                        gray[ix1, iy1] = 0
                        kb = kb + 1
                    kj = kj + 1

            cv2.imshow("gray1", gray)  # 显示二值化图像

            print("总像素点数", kj)
            print("ka", ka)
            print("kb", kb)

            for ix2 in range(ymin, ymax+1):
                for iy2 in range(xmin, xmax+1):
                    hj = float(img1[ix2][iy2])
                    p2.append(hj)
                    if(hj>=rav-10 and hj<= rav+10):
                        gray[ix2, iy2] = 255
                        kc = kc + 1
                    else:
                        gray[ix2, iy2] = 0
                        kd = kd + 1

            print("kc", kc)
            print("kd", kd)
            # print("深度统计", p2)

            cv2.imshow("gray2", gray)  # 显示二值化图像

            contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lunkou = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
            cv2.imshow('lunkou', lunkou)
            cv2.imwrite('lunkou.jpg', lunkou)

            num = len(contours)
            for i in range(num):
                if len(contours[i]) > 100:
                    xsd = cv2.contourArea(contours[i])
                    print("**************目标像素点数为**************")
                    print(xsd)
                    x, y, w1, h1 = cv2.boundingRect(contours[i])
                    cv2.rectangle(img, (x, y), (x + w1, y + h1), (255, 255, 255), 2)
                    cv2.imshow("result1", img)
                    print("**************目标框信息**************")
                    print(x, y, w1, h1)
                    x1 = x + w1
                    y1 = y + h1

                    xmi = x - 3
                    xma = x + 3
                    ymi = y - 3
                    yma = y + 3

                    x1mi = x1 - 3
                    x1ma = x1 + 3
                    y1mi = y1 - 3
                    y1ma = y1 + 3

                    h11 = float(img1[y][x])
                    h21 = float(img1[y1][x1])

                    # for xk in range(xmi, xma):
                    #     for yk in range(ymi, yma):
                    #         h_k = float(img1[yk][xk])
                    #         if (h_k > 10):
                    #             sum1 = sum1 + h_k
                    #             ke = ke + 1
                    #
                    # print("ke", ke)
                    # h11 = int(sum1 / ke);
                    #
                    # for xk1 in range(x1mi, x1ma):
                    #     for yk1 in range(y1mi, y1ma):
                    #         h_k1 = float(img1[yk1][xk1])
                    #         if (h_k1 > 10):
                    #             sum2 = sum2 + h_k1
                    #             kf = kf + 1
                    #
                    # print("kf", kf)
                    # h21 = int(sum2 / kf);


                    print("h11", h11)
                    print("h21", h21)

                    p1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], h11)

                    p2 = rs.rs2_deproject_pixel_to_point(depth_intrin, [x1, y1], h21)

                    print("p1", p1)
                    print("p2", p2)

                    result = abs((p2[0] - p1[0]) * (p2[1] - p1[1])) / 100
                    print("**************矩形框面积为**************")
                    print(result)
                    result1 = result * xsd / (w * h)
                    print("**************叶面积为**************")
                    print(result1)



            cv2.waitKey()  # 按下任意按键 才动*****
            cv2.destroyAllWindows()  # 释放所有窗体****








