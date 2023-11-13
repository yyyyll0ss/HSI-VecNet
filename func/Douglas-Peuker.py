import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os.path as osp
import os
from Hyper import get_dataset_rgb,color_map
from Pytorch import generate_coco_ann_DP, generate_coco_mask
import json


def save_per_class_result(per_class_result,output_dir,dataset_name,class_num):
    """
    save different result to .png
    """
    file_path = osp.join(output_dir, dataset_name)
    if not osp.exists(file_path):
        os.makedirs(file_path)
    cv2.imwrite(osp.join(file_path,f'poly_class{class_num}.png'),per_class_result)

def draw_poly_result(all_class_polygon,save_path,dataset):
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'gray', 'indianred',
    #           'chocolate', 'tan', 'skyblue', 'olive', 'lime', 'teal', 'hotpink', 'purple','k']
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
    #           [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
    #           [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [238, 0, 0]]
    result = cv2.imread(f'../{dataset}_rgb.png')
    #result = np.zeros((img.shape[0], img.shape[1], 3)).astype('uint8')
    colors = color_map(dataset).tolist()
    for key,values in all_class_polygon.items():
        for polygon in values:
            polygon = np.array(polygon,np.int32)
            cv2.polylines(result,[polygon],True,colors[int(key)-1][::-1],1)
    file_path = osp.join(save_path,dataset)
    if not osp.exists(file_path):
        os.makedirs(file_path)

    cv2.imwrite(osp.join(file_path, f'{dataset}-polygon-result.png'), result)

def draw_polygon(all_class_polygon,save_path,dataset,numTrain):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as Patches
    import os.path as osp
    import os
    from skimage import io

    colors = color_map(dataset).tolist()
    image = io.imread(f'/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/{dataset}_rgb.png')
    plt.axis('off')
    plt.imshow(image)

    for key,values in all_class_polygon.items():
        for polygon in values:
            polygon = np.array(polygon)
            color = np.array(colors[int(key)-1])/255
            plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=0.5))
            plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.', linewidth=0.75,markersize=2)   # linewidth=2,markersize=3.5

    file_path = osp.join(save_path, dataset)
    if not osp.exists(file_path):
        os.makedirs(file_path)
    impath = osp.join(file_path, f'{dataset}_{numTrain}_polygon_result.pdf')

    plt.savefig(impath, bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.clf()




if __name__ == "__main__":
    # load image data
    dataset = ['indian_pines','WHU_Hi_HongHu','WHU_Hi_LongKou','salinas_valley','PaviaU','Houston']
    numTrain = 0.2
    dataset_ = dataset[5]
    #get dataset rgb
    #get_dataset_rgb(dataset[2])

    img = np.load(f'/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/pictures/{dataset_}_{numTrain}_result.npy')   ######
    mask_output_dir = '/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/per_class_label'
    countour_output_dir = '/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/per_class_countour'
    polygon_output_dir = '/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/per_class_polygon'
    all_result_output_dir = '/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/polygon_result'

    all_class_polygons = {}

    sum_contour = 0
    sum_poly = 0
    for class_num in range(1,img.max() + 1):
        per_class_mask = np.zeros((img.shape[0],img.shape[1],1))
        #print(np.sum(img == class_num))
        if np.sum(img == class_num) == 0:
            continue
        per_class_mask[np.where(img == class_num)] = 1
        per_class_mask = per_class_mask.astype('uint8') * 255

        #save 0-255 label mask
        save_per_class_result(per_class_mask,mask_output_dir, f'{dataset_}',class_num)   #####

        #save counter
        _, thresh = cv2.threshold(per_class_mask * 255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        per_class_mask_bgr = cv2.cvtColor(per_class_mask,cv2.COLOR_GRAY2BGR)
        Counter_res = cv2.drawContours(per_class_mask_bgr,contours,-1,(0,255,0),1)
        save_per_class_result(Counter_res,countour_output_dir,f'{dataset_}',class_num)   #####

        per_class_polygons = []

        for contour in contours:
            if cv2.contourArea(contour) < 20:   # indian,salinas(40),HongHu,LongKou(100)
                continue
            if len(contour) < 3:
                continue
            # 2.进行多边形逼近，得到多边形的角点
            epsilon = 0.037 * cv2.arcLength(contour,True)   #0.05 0.035
            print('countour points:',len(contour))
            sum_contour += len(contour)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            print('approx points:', len(approx))
            sum_poly += len(approx)
            # 3.画出多边形
            cv2.polylines(per_class_mask_bgr, [approx], True, (0, 255, 0), 1)
            if len(approx) > 2:
                approx = approx.squeeze().tolist()
                per_class_polygons.append(approx)
        #save polygon
        save_per_class_result(per_class_mask_bgr,polygon_output_dir,f'{dataset_}',class_num)   #####

        all_class_polygons[str(class_num)] = per_class_polygons

    draw_polygon(all_class_polygons,all_result_output_dir,f'{dataset_}',numTrain=0.2)   #####
    print('sum_contour:',sum_contour)
    print('sum_poly:',sum_poly)

    #get DP coco ann results
    image_result = generate_coco_ann_DP(all_class_polygons, 0)

    # creat polygon save path
    poly_path_ = f'../output/{dataset_}'   #####
    if not osp.exists(poly_path_):
        os.makedirs(poly_path_)
    poly_path = osp.join(poly_path_, f'{dataset_}_{numTrain}_DP.json')   #####

    # save two style result to json file
    with open(poly_path, 'w') as _out:
        json.dump(image_result, _out)














