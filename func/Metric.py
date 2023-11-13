import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
from shapely import geometry
from shapely.geometry import Polygon
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import csv

import shapely
import shapely.geometry
import shapely.affinity
import shapely.ops
import shapely.prepared
import shapely.validation

import random
import multiprocess
from multiprocess import Pool
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection

def bounding_box(points):
    """returns a list containing the bottom left and the top right
    points in the sequence
    Here, we traverse the collection of points only once,
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float('inf'), float('inf')
    top_right_x, top_right_y = float('-inf'), float('-inf')
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [bot_left_x, bot_left_y, top_right_x - bot_left_x, top_right_y - bot_left_y]

def compare_polys(poly_a, poly_b):
    """Compares two polygons via the "polis" distance metric.
    See "A Metric for Polygon Comparison and Building Extraction
    Evaluation" by J. Avbelj, et al.
    Input:
        poly_a: A Shapely polygon.
        poly_b: Another Shapely polygon.
    Returns:
        The "polis" distance between these two polygons.
    """
    bndry_a, bndry_b = poly_a.exterior, poly_b.exterior
    dist = polis(bndry_a.coords, bndry_b)
    dist += polis(bndry_b.coords, bndry_a)
    return dist


def polis(coords, bndry):
    """Computes one side of the "polis" metric.
    Input:
        coords: A Shapley coordinate sequence (presumably the vertices
                of a polygon).
        bndry: A Shapely linestring (presumably the boundary of
        another polygon).

    Returns:
        The "polis" metric for this pair.  You usually compute this in
        both directions to preserve symmetry.
    """
    sum = 0.0
    for pt in (geometry.Point(c) for c in coords[:-1]): # Skip the last point (same as first)
        sum += bndry.distance(pt)
    return sum /float( 2 *len(coords))

class PolisEval():

    def __init__(self, cocoGt=None, cocoDt=None):
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.evalImgs = defaultdict(list)
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.stats = []
        self.imgIds = list(sorted(self.cocoGt.imgs.keys()))

    def _prepare(self):
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.imgIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=self.imgIds))
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id']].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluateImg(self, imgId):
        gts = self._gts[imgId]
        dts = self._dts[imgId]

        if len(gts) == 0 or len(dts) == 0:
            return 0

        gt_bboxs = [bounding_box(np.array(gt['segmentation'][0]).reshape(-1, 2)) for gt in gts if
                    len(gt['segmentation']) != 0]
        dt_bboxs = [bounding_box(np.array(dt['segmentation'][0]).reshape(-1, 2)) for dt in dts if
                    len(dt['segmentation']) != 0]
        gt_polygons = [np.array(gt['segmentation'][0]).reshape(-1, 2) for gt in gts if len(gt['segmentation']) != 0]
        dt_polygons = [np.array(dt['segmentation'][0]).reshape(-1, 2) for dt in dts if len(dt['segmentation']) != 0]

        # IoU match
        iscrowd = [0] * len(gt_bboxs)
        # ious = maskUtils.iou(gt_bboxs, dt_bboxs, iscrowd)
        ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)

        # compute polis
        img_polis_avg = 0
        num_sample = 0
        for i, gt_poly in enumerate(gt_polygons):
            matched_idx = np.argmax(ious[:, i])
            iou = ious[matched_idx, i]
            if iou > 0.5:  # iouThres:
                polis = compare_polys(Polygon(gt_poly), Polygon(dt_polygons[matched_idx]))
                img_polis_avg += polis
                num_sample += 1

        if num_sample == 0:
            return 0
        else:
            return img_polis_avg / num_sample

    def evaluate(self):
        self._prepare()
        polis_tot = 0

        num_valid_imgs = 0
        for imgId in tqdm(self.imgIds):
            img_polis_avg = self.evaluateImg(imgId)

            if img_polis_avg == 0:
                continue
            else:
                polis_tot += img_polis_avg

                num_valid_imgs += 1

        polis_avg = polis_tot / num_valid_imgs

        print('average polis: %f' % (polis_avg))

        return polis_avg





def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    is_zero = I == 0
    if is_void:
        return 1.0
    if is_zero:
        return 0
    else:
        return iou

def compute_IoU_cIoU(input_json, gti_annotations, is_DP):
    gt_coco = COCO(gti_annotations)
    dt_coco = gt_coco.loadRes(input_json)

    image_ids = gt_coco.getImgIds()
    cat_ids = dt_coco.getCatIds()

    # per_class_ann = dt_coco.getAnnIds(catIds=cat_ids[0])
    # print(per_class_ann)
    #image_ids = coco.getImgIds(catIds=coco.getCatIds())

    bar = tqdm(image_ids)

    for image_id in bar:

        img = dt_coco.loadImgs(image_id)[0]
        list_iou = []
        list_ciou = []
        pss = []
        for class_num in cat_ids:
            annotation_ids = dt_coco.getAnnIds(catIds=cat_ids[class_num-1])
            if len(annotation_ids) == 0:
                list_iou.append(0)
                list_ciou.append(0)
                continue
            annotations = dt_coco.loadAnns(annotation_ids)

            N = 0
            for _idx, annotation in enumerate(annotations):
                try:
                    rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
                except Exception:
                    import ipdb; ipdb.set_trace()
                m = cocomask.decode(rle)
                if _idx == 0:
                    mask = m.reshape((img['height'], img['width']))
                    N = len(annotation['segmentation'][0]) // 2
                else:
                    mask = mask + m.reshape((img['height'], img['width']))
                    N = N + len(annotation['segmentation'][0]) // 2

            mask = mask != 0


            # annotation_ids = gt_coco.getAnnIds(imgIds=img['id'])
            # annotations = gt_coco.loadAnns(annotation_ids)
            annotation_ids = gt_coco.getAnnIds(catIds=cat_ids[class_num-1])
            annotations = gt_coco.loadAnns(annotation_ids)
            N_GT = 0
            for _idx, annotation in enumerate(annotations):
                rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
                m = cocomask.decode(rle)
                if _idx == 0:
                    mask_gti = m.reshape((img['height'], img['width']))
                    N_GT = len(annotation['segmentation'][0]) // 2
                else:
                    mask_gti = mask_gti + m.reshape((img['height'], img['width']))
                    N_GT = N_GT + len(annotation['segmentation'][0]) // 2

            mask_gti = mask_gti != 0

            ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
            iou = calc_IoU(mask, mask_gti)
            list_iou.append(iou)
            list_ciou.append(iou * ps)
            pss.append(ps)

            # bar.set_description("class:%2.0f, iou: %2.4f, c-iou: %2.4f, ps:%2.4f" % (class_num, iou, iou * ps, ps))
            # bar.refresh()
    print("Done!")
    print("IoU list:",list_iou)
    print("C-IoU list:",list_ciou)
    if is_DP == False:
        with open('../iou_ciou.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list_iou)
            writer.writerow(list_ciou)
    if is_DP == True:
        with open('../iou_ciou_DP.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list_iou)
            writer.writerow(list_ciou)
    print("Mean IoU: ", np.sum(list_iou)/len(cat_ids))
    print("Mean C-IoU: ", np.sum(list_ciou)/len(cat_ids))
    return np.sum(list_iou)/len(cat_ids), np.sum(list_ciou)/len(cat_ids)



class ContourEval:
    def __init__(self, coco_gt, coco_dt):
        """
        @param coco_gt: coco object with ground truth annotations
        @param coco_dt: coco object with detection results
        """
        self.coco_gt = coco_gt  # ground truth COCO API
        self.coco_dt = coco_dt  # detections COCO API

        self.img_ids = sorted(coco_gt.getImgIds())
        self.cat_ids = sorted(coco_dt.getCatIds())

    def evaluate(self, pool=None):

        gts = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=self.img_ids))
        dts = self.coco_dt.loadAnns(self.coco_dt.getAnnIds(imgIds=self.img_ids))

        _gts = defaultdict(list)  # gt for evaluation
        _dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            _gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            _dts[dt['image_id'], dt['category_id']].append(dt)
        evalImgs = defaultdict(list)  # per-image per-category evaluation results

        # Compute metric
        args_list = []
        # i = 1000
        for img_id in self.img_ids:
            for cat_id in self.cat_ids:
                gts = _gts[img_id, cat_id]
                dts = _dts[img_id, cat_id]
                args_list.append((gts, dts))
                # i -= 1
            # if i <= 0:
            #     break

        if pool is None:
            measures_list = []
            for args in tqdm(args_list, desc="Contour metrics"):
                measures_list.append(compute_contour_metrics(args))
        else:
            measures_list = list(tqdm(pool.imap(compute_contour_metrics, args_list), desc="Contour metrics", total=len(args_list)))
        measures_list = [measure for measures in measures_list for measure in measures]  # Flatten list
        # half_tangent_cosine_similarities_list, edge_distances_list = zip(*measures_list)
        # half_tangent_cosine_similarities_list = [item for item in half_tangent_cosine_similarities_list if item is not None]
        measures_list = [value for value in measures_list if value is not None]
        max_angle_diffs = np.array(measures_list)
        max_angle_diffs = max_angle_diffs * 180 / np.pi  # Convert to degrees

        return max_angle_diffs

def compute_contour_metrics(gts_dts):
    gts, dts = gts_dts
    gt_polygons = [shapely.geometry.Polygon(np.array(coords).reshape(-1, 2)) for ann in gts
                   for coords in ann["segmentation"]]
    dt_polygons = [shapely.geometry.Polygon(np.array(coords).reshape(-1, 2)) for ann in dts
                   for coords in ann["segmentation"]]
    fixed_gt_polygons = fix_polygons(gt_polygons, buffer=0.0001)  # Buffer adds vertices but is needed to repair some geometries
    fixed_dt_polygons = fix_polygons(dt_polygons)
    # cosine_similarities, edge_distances = \
    #     polygon_utils.compute_polygon_contour_measures(dt_polygons, gt_polygons, sampling_spacing=2.0, min_precision=0.5,
    #                                                    max_stretch=2)
    max_angle_diffs = compute_polygon_contour_measures(fixed_dt_polygons, fixed_gt_polygons, sampling_spacing=2.0, min_precision=0.5, max_stretch=2)

    return max_angle_diffs

def compute_polygon_contour_measures(pred_polygons: list, gt_polygons: list, sampling_spacing: float, min_precision: float, max_stretch: float, metric_name: str="cosine", progressbar=False):
    """
    pred_polygons are sampled with sampling_spacing before projecting those sampled points to gt_polygons.
    Then the
    @param pred_polygons:
    @param gt_polygons:
    @param sampling_spacing:
    @param min_precision: Polygons in pred_polygons must have a precision with gt_polygons above min_precision to be included in further computations
    @param max_stretch:  Exclude edges that have been stretched by the projection more than max_stretch from further computation
    @param metric_name: Metric type, can be "cosine" or ...
    @return:
    """
    assert isinstance(pred_polygons, list), "pred_polygons should be a list"
    assert isinstance(gt_polygons, list), "gt_polygons should be a list"
    if len(pred_polygons) == 0 or len(gt_polygons) == 0:
        return np.array([]), [], []
    assert isinstance(pred_polygons[0], shapely.geometry.Polygon), \
        f"Items of pred_polygons should be of type shapely.geometry.Polygon, not {type(pred_polygons[0])}"
    assert isinstance(gt_polygons[0], shapely.geometry.Polygon), \
        f"Items of gt_polygons should be of type shapely.geometry.Polygon, not {type(gt_polygons[0])}"
    gt_polygons = shapely.geometry.collection.GeometryCollection(gt_polygons)
    pred_polygons = shapely.geometry.collection.GeometryCollection(pred_polygons)
    # Filter pred_polygons to have at least a precision with gt_polygons of min_precision
    filtered_pred_polygons = [pred_polygon for pred_polygon in pred_polygons if min_precision < pred_polygon.intersection(gt_polygons).area / pred_polygon.area]
    # Extract contours of gt polygons
    gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons for contour in [polygon.exterior, *polygon.interiors]])
    # Measure metric for each pred polygon
    if progressbar:
        process_id = int(multiprocess.current_process().name[-1])
        iterator = tqdm(filtered_pred_polygons, desc="Contour measure", leave=False, position=process_id)
    else:
        iterator = filtered_pred_polygons
    half_tangent_max_angles = [compute_contour_measure(pred_polygon, gt_contours, sampling_spacing=sampling_spacing, max_stretch=max_stretch, metric_name=metric_name)
                               for pred_polygon in iterator]
    return half_tangent_max_angles

def fix_polygons(polygons, buffer=0.0):
    polygons = [
        geom if geom.is_valid else geom.buffer(0) for geom in polygons
    ]
    polygons_geom = shapely.ops.unary_union(polygons)  # Fix overlapping polygons
    polygons_geom = polygons_geom.buffer(buffer)  # Fix self-intersecting polygons and other things
    fixed_polygons = []
    if polygons_geom.geom_type == "MultiPolygon":
        for poly in polygons_geom:
            fixed_polygons.append(poly)
    elif polygons_geom.geom_type == "Polygon":
        fixed_polygons.append(polygons_geom)
    else:
        raise TypeError(f"Geom type {polygons_geom.geom_type} not recognized.")
    return fixed_polygons

def compute_contour_measure(pred_polygon, gt_contours, sampling_spacing, max_stretch, metric_name="cosine"):
    pred_contours = shapely.geometry.GeometryCollection([pred_polygon.exterior, *pred_polygon.interiors])
    sampled_pred_contours = sample_geometry(pred_contours, sampling_spacing)
    # Project sampled contour points to ground truth contours
    projected_pred_contours = project_onto_geometry(sampled_pred_contours, gt_contours)
    contour_measures = []
    for contour, proj_contour in zip(sampled_pred_contours, projected_pred_contours):
        coords = np.array(contour.coords[:])
        proj_coords = np.array(proj_contour.coords[:])
        edges = coords[1:] - coords[:-1]
        proj_edges = proj_coords[1:] - proj_coords[:-1]
        # Remove edges with a norm of zero
        edge_norms = np.linalg.norm(edges, axis=1)
        proj_edge_norms = np.linalg.norm(proj_edges, axis=1)
        norm_valid_mask = 0 < edge_norms * proj_edge_norms
        edges = edges[norm_valid_mask]
        proj_edges = proj_edges[norm_valid_mask]
        edge_norms = edge_norms[norm_valid_mask]
        proj_edge_norms = proj_edge_norms[norm_valid_mask]
        # Remove edge that have stretched more than max_stretch (invalid projection)
        stretch = edge_norms / proj_edge_norms
        stretch_valid_mask = np.logical_and(1 / max_stretch < stretch, stretch < max_stretch)
        edges = edges[stretch_valid_mask]
        if edges.shape[0] == 0:
            # Invalid projection for the whole contour, skip it
            continue
        proj_edges = proj_edges[stretch_valid_mask]
        edge_norms = edge_norms[stretch_valid_mask]
        proj_edge_norms = proj_edge_norms[stretch_valid_mask]
        scalar_products = np.abs(np.sum(np.multiply(edges, proj_edges), axis=1) / (edge_norms * proj_edge_norms))
        try:
            contour_measures.append(scalar_products.min())
        except ValueError:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
            ax = axes.ravel()
            plot_geometries(ax[0], [contour])
            plot_geometries(ax[1], [proj_contour])
            plot_geometries(ax[2], gt_contours)
            fig.tight_layout()
            plt.show()
    if len(contour_measures):
        min_scalar_product = min(contour_measures)
        measure = np.arccos(min_scalar_product)
        return measure
    else:
        return None

def sample_geometry(geom, density):
    """
    Sample edges of geom with a homogeneous density.
    @param geom:
    @param density:
    @return:
    """
    if isinstance(geom, shapely.geometry.GeometryCollection):
        # tic = time.time()

        sampled_geom = shapely.geometry.GeometryCollection([sample_geometry(g, density) for g in geom])

        # toc = time.time()
        # print(f"sample_geometry: {toc - tic}s")
    elif isinstance(geom, shapely.geometry.Polygon):
        sampled_exterior = sample_geometry(geom.exterior, density)
        sampled_interiors = [sample_geometry(interior, density) for interior in geom.interiors]
        sampled_geom = shapely.geometry.Polygon(sampled_exterior, sampled_interiors)
    elif isinstance(geom, shapely.geometry.LineString):
        sampled_x = []
        sampled_y = []
        coords = np.array(geom.coords[:])
        lengths = np.linalg.norm(coords[:-1] - coords[1:], axis=1)
        for i in range(len(lengths)):
            start = geom.coords[i]
            end = geom.coords[i + 1]
            length = lengths[i]
            num = max(1, int(round(length / density))) + 1
            x_seq = np.linspace(start[0], end[0], num)
            y_seq = np.linspace(start[1], end[1], num)
            if 0 < i:
                x_seq = x_seq[1:]
                y_seq = y_seq[1:]
            sampled_x.append(x_seq)
            sampled_y.append(y_seq)
        sampled_x = np.concatenate(sampled_x)
        sampled_y = np.concatenate(sampled_y)
        sampled_coords = zip(sampled_x, sampled_y)
        sampled_geom = shapely.geometry.LineString(sampled_coords)
    else:
        raise TypeError(f"geom of type {type(geom)} not supported!")
    return sampled_geom

def project_onto_geometry(geom, target, pool: Pool=None):
    """
    Projects all points from line_string onto target.
    @param geom:
    @param target:
    @param pool:
    @return:
    """
    if isinstance(geom, shapely.geometry.GeometryCollection):
        # tic = time.time()

        if pool is None:
            projected_geom = [project_onto_geometry(g, target, pool=pool) for g in geom]
        else:
            partial_project_onto_geometry = partial(project_onto_geometry, target=target)
            projected_geom = pool.map(partial_project_onto_geometry, geom)
        projected_geom = shapely.geometry.GeometryCollection(projected_geom)

        # toc = time.time()
        # print(f"project_onto_geometry: {toc - tic}s")
    elif isinstance(geom, shapely.geometry.Polygon):
        projected_exterior = project_onto_geometry(geom.exterior, target)
        projected_interiors = [project_onto_geometry(interior, target) for interior in geom.interiors]
        try:
            projected_geom = shapely.geometry.Polygon(projected_exterior, projected_interiors)
        except shapely.errors.TopologicalError as e:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
            ax = axes.ravel()
            plot_geometries(ax[0], [geom])
            plot_geometries(ax[1], target)
            plot_geometries(ax[2], [projected_exterior, *projected_interiors])
            fig.tight_layout()
            plt.show()
            raise e
    elif isinstance(geom, shapely.geometry.LineString):
        projected_coords = [point_project_onto_geometry(coord, target) for coord in geom.coords]
        projected_geom = shapely.geometry.LineString(projected_coords)
    else:
        raise TypeError(f"geom of type {type(geom)} not supported!")
    return projected_geom

def point_project_onto_geometry(coord, target):
    point = shapely.geometry.Point(coord)
    _, projected_point = shapely.ops.nearest_points(point, target)
    # dist = point.distance(projected_point)
    return projected_point.coords[0]

def plot_geometries(axis, geometries, linewidths=1, markersize=3):
    if len(geometries):
        patches = []
        for i, geometry in enumerate(geometries):
            if geometry.geom_type == "Polygon":
                polygon = shapely.geometry.Polygon(geometry)
                if not polygon.is_empty:
                    patch = PolygonPatch(polygon)
                    patches.append(patch)
                axis.plot(*polygon.exterior.xy, marker="o", markersize=markersize)
                for interior in polygon.interiors:
                    axis.plot(*interior.xy, marker="o", markersize=markersize)
            elif geometry.geom_type == "LineString" or geometry.geom_type == "LinearRing":
                axis.plot(*geometry.xy, marker="o", markersize=markersize)
            else:
                raise NotImplementedError(f"Geom type {geometry.geom_type} not recognized.")
        random.seed(1)
        colors = random.choices([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0.5, 1, 0, 1],
            [1, 0.5, 0, 1],
            [0.5, 0, 1, 1],
            [1, 0, 0.5, 1],
            [0, 0.5, 1, 1],
            [0, 1, 0.5, 1],
        ], k=len(patches))
        edgecolors = np.array(colors)
        facecolors = edgecolors.copy()
        p = PatchCollection(patches, facecolors=facecolors, edgecolors=edgecolors, linewidths=linewidths)
        axis.add_collection(p)