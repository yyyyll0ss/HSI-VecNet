import shapefile
from shapely.geometry import Point,Polygon,MultiPoint
import os
import json
from matplotlib import pyplot as plt
from Hyper import matDataLoad
import mmcv
import os.path as osp
import io
import scipy.io
"""
    该脚本负责将envi矢量标注产生的shp文件，进行地理坐标变换，获取外接正交矩形，将shp文件中的信息存入
一定格式的json文件中。
"""

#获取外接正交矩形函数
def get_rotated_rectangle(polygon):
    new_polygon = list(polygon.exterior.coords)
    x, y = zip(*new_polygon)  #unzip
    min_x,max_x,min_y,max_y = min(x),max(x),min(y),max(y)
    return [min_x,min_y,max_x,max_y]

def get_shape_list(path):

    file = shapefile.Reader(path)#读取
    records = file.records()  #列表信息（id，pic_num）
    borders = file.shapes()  #边界信息（polygon）

    shape_list = []
    for i in range(len(records)):
        #坐标变换
        points = borders[i].points  #该对象的所有轮廓点
        #求轮廓外接矩形
        polygon = Polygon(points)
        bbox = get_rotated_rectangle(polygon)

    #------------可视化----------------

        # x, y = zip(*real_points)  #unzip
        # fig, ax = plt.subplots()  # 生成一张图和一张子图
        # plt.plot(x, y, color='#6666ff', label='fungis')  # x横坐标 y纵坐标 ‘k-’线性为黑色
        # rect = plt.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill = False,edgecolor = 'red',linewidth = 1)
        # ax.add_patch(rect)
        # ax.grid()  # 添加网格线
        # ax.axis('equal')
        # ax = plt.gca()                         #获取到当前坐标轴信息
        # ax.xaxis.set_ticks_position('top')     #将X坐标轴移到上面
        # ax.invert_yaxis()                      #反转Y坐标轴
        # plt.show()

        instance_dic = dict(class_num=records[i][1],boxes=bbox, points=points)
        shape_list.append(instance_dic)
    return shape_list

def convert_json_to_coco_style(label_dir, dataset, out_file):
    obj_count = 0
    images = []
    annotations = []
    prev_dict = mmcv.load(label_dir)
    data = scipy.io.loadmat(f'../datasets/{dataset}/' + dataset + '.mat')
    height, width = data[list(data.keys())[-1]].shape[:2]
    #height, width = matDataLoad('../datasets/' + dataset + '.mat').shape[:2]
    images.append(dict(
        id = 0,
        file_name = dataset+'.mat',
        height = height,
        width = width))
    for item in prev_dict['shape']:
        x_min, y_min, x_max, y_max = item['boxes']
        mask = item['points']
        if mask != None:
            px = [item[0] for item in mask]
            py = [item[1] for item in mask]
            poly = [(x , y ) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
        else:
            poly = None

        data_anno = dict(
            image_id=0,
            id=obj_count,
            category_id=int(item['class_num'].replace('#', '').split(' ')[0]),
            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
            area=(x_max - x_min) * (y_max - y_min),
            segmentation=[poly],
            iscrowd=0)
        annotations.append(data_anno)
        obj_count += 1
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        # indian_pines
        # categories=[{'id':1, 'name': 'Alfalfa'},{'id':2, 'name': 'Corn-notill'},{'id':3, 'name': 'Corn-mintill'},{'id':4, 'name': 'Corn'},
        #             {'id':5, 'name': 'Grass-pasture'},{'id':6, 'name': 'Grass-trees'},{'id':7, 'name': 'Grass-pasture-mowed'},{'id':8, 'name': 'Hay-windrowed'},
        #             {'id':9, 'name': 'Oats'},{'id':10, 'name': 'Soybean-nottill'},{'id':11, 'name': 'Soybean-mintill'},{'id':12, 'name': 'Soymean-clean'},
        #             {'id':13, 'name': 'Wheat'},{'id':14, 'name': 'Woods'},{'id':15, 'name': 'Buildings-Grass-Trees-Drives'},{'id':16, 'name': 'Stone-Steel-Towers'}])
        # salinas_valley
        # categories=[{'id':1, 'name': 'Brocoli_green_weed_1'},{'id':2, 'name': 'Brocoli_green_weed_22'},{'id':3, 'name': 'Fallow'},{'id':4, 'name': 'Fallow_rough_plow'},
        #             {'id':5, 'name': 'Fallow_smooth'},{'id':6, 'name': 'Stubble'},{'id':7, 'name': 'Celery'},{'id':8, 'name': 'Grapes_untrained'},
        #             {'id':9, 'name': 'Soil_vinyard_develop'},{'id':10, 'name': 'Corn_senesced_green_weeds'},{'id':11, 'name': 'Lettuce_romaine_4wk'},{'id':12, 'name': 'Lettuce_romaine_5wk'},
        #             {'id':13, 'name': 'Lettuce_romaine_6wk'},{'id':14, 'name': 'Lettuce_romaine_7wk'},{'id':15, 'name': 'Vinyard_untrained'},{'id':16, 'name': 'Vinyard_vertical_trellis'}])
        # WHU_Hi_HongHu
        # categories=[{'id':1, 'name': 'Red_roof'},{'id':2, 'name': 'Road'},{'id':3, 'name': 'Bare_soil'},{'id':4, 'name': 'Cotton'},
        #             {'id':5, 'name': 'Cotton_firewood'},{'id':6, 'name': 'Rape'},{'id':7, 'name': 'Chinese_cabbage'},{'id':8, 'name': 'Pakchoi'},
        #             {'id':9, 'name': 'Cabbage'},{'id':10, 'name': 'Tuber_mustard'},{'id':11, 'name': 'Brassica_parachinensis'},{'id':12, 'name': 'Brassica_chinensis'},
        #             {'id':13, 'name': 'Small_Brassica_chinensis'},{'id':14, 'name': 'Lactuca_sativa'},{'id':15, 'name': 'Celtuce'},{'id':16, 'name': 'Film_covered_lettuce'},
        #             {'id':17, 'name': 'Romaine_lettuce'},{'id':18, 'name': 'Carrot'},{'id':19, 'name': 'White_radish'},{'id':20, 'name': 'Garlic_sprout'},
        #             {'id':21, 'name': 'Broad_bean'},{'id':22, 'name': 'Tree'}])
        # WHU_Hi_LongKou
        categories = [{'id': 1, 'name': 'Corn'}, {'id': 2, 'name': 'Cotton'}, {'id': 3, 'name': 'Sesame'},
                      {'id': 4, 'name': 'Broad-leaf soybean'},{'id': 5, 'name': 'Narrow-leaf soybean'}, {'id': 6, 'name': 'Rice'},
                      {'id': 7, 'name': 'Water'},{'id': 8, 'name': 'Roads and houses'},{'id': 9, 'name': 'Mixed weed'}])

    with open(out_file+f'/{dataset}_coco.json','w') as f:
        json.dump(coco_format_json,f)
        print('格式转换完成')


if __name__ == "__main__":
    #---------------------------将每张图片的标注信息存入json-----------------------------
    path = r'/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/datasets/Houston/Houston_label.shp'
    shape_list = get_shape_list(path)
    label_path = r'/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/datasets/Houston'

    pic_label_dict = dict(shape=shape_list, imagePath=f'Houston.mat')
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    with open(os.path.join(label_path, f'Houston.json'),'w') as f:
        json.dump(pic_label_dict,f)
        print(f"图片注释保存成功，生成json文件")

    json_path = '/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/datasets/Houston/Houston.json'
    convert_json_to_coco_style(json_path, 'Houston', label_path)