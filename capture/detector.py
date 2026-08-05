"""Animal bounding boxes, via detectron2.

Inlined from AniMer's `demo_video.py`, which is otherwise not needed here.
Importing that module pulled in trimesh and pyrender for renderer classes
stage 1 never calls; these twenty lines are the only part it used.
"""
import numpy as np

# COCO contiguous ids for cat, dog, horse, sheep, cow, bear, zebra
ANIMAL_CLASSES = [15, 16, 17, 18, 19, 21, 22]


def build_detector(score_thresh: float):
    import detectron2.config
    import detectron2.engine
    from detectron2 import model_zoo

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = ("https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/"
                         "faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl")
    return detectron2.engine.DefaultPredictor(cfg)


def detect_animals(detector, img_rgb, score_thresh, max_animals):
    instances = detector(img_rgb)['instances']
    keep = [i for i, (c, s) in enumerate(zip(instances.pred_classes, instances.scores))
            if (int(c) in ANIMAL_CLASSES) and (float(s) > score_thresh)]
    if not keep:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = instances.pred_boxes.tensor[keep].cpu().numpy()
    scores = instances.scores[keep].cpu().numpy()
    if max_animals > 0 and len(boxes) > max_animals:
        boxes = boxes[np.argsort(-scores)[:max_animals]]
    return boxes
