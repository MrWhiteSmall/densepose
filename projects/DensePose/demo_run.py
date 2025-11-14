from typing import Any, ClassVar, Dict
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    create_extractor,
)


VISUALIZERS: ClassVar[Dict[str, object]] = {
    "dp_contour": DensePoseResultsContourVisualizer,
    "dp_segm": DensePoseResultsFineSegmentationVisualizer,
    "dp_u": DensePoseResultsUVisualizer,
    "dp_v": DensePoseResultsVVisualizer,
    "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
    "dp_cse_texture": DensePoseOutputsTextureVisualizer,
    "dp_vertex": DensePoseOutputsVertexVisualizer,
    "bbox": ScoredBoundingBoxVisualizer,
}


def get_dp_predictor():
    config_fpath = 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'
    model_fpath = 'model_final_162be9.pkl'
    opts = []
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def get_outputs_by_predictor(dp_predictor, img_pil):
    import numpy as np
    import cv2
    img_np = np.array(img_pil)
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        outputs = dp_predictor(img)["instances"]

    return outputs

def create_context(cfg) -> Dict[str, Any]:
    visualizers = []
    extractors = []
    texture_atlas = get_texture_atlas(None)
    texture_atlases_dict = get_texture_atlases(None)
    vis = VISUALIZERS['dp_segm'](
        cfg=cfg,
        texture_atlas=texture_atlas,
        texture_atlases_dict=texture_atlases_dict,
    )
    visualizers.append(vis)
    extractor = create_extractor(vis)
    extractors.append(extractor)
    visualizer = CompoundVisualizer(visualizers)
    extractor = CompoundExtractor(extractors)
    context = {
        "extractor": extractor,
        "visualizer": visualizer,
        "entry_idx": 0,
    }
    return context


def execute_on_outputs(
    context: Dict[str, Any], outputs, img_pil, 
):
    import cv2
    import numpy as np

    visualizer = context["visualizer"]
    extractor = context["extractor"]
    img_np = np.array(img_pil)
    image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    bimg = np.zeros_like(image)
    data = extractor(outputs)
    image_vis = visualizer.visualize(image, data) # (1280, 1120, 3)
    bimage_vis = visualizer.visualize(bimg, data)

    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
    bimage_vis = cv2.cvtColor(bimage_vis, cv2.COLOR_BGR2RGB)

    return image_vis, bimage_vis



def main():
    dp_cfg, dp_predictor = get_dp_predictor()

    from PIL import Image
    img_path = 'image.jpg'
    img_pil = Image.open(img_path).convert('RGB') 

    dp_outputs = get_outputs_by_predictor(dp_predictor, img_pil)
    
    dp_context = create_context(dp_cfg)

    res_bg_dp, res_nobg_dp = execute_on_outputs(dp_context, dp_outputs, img_pil)

    res_bg_dp_pil = Image.fromarray(res_bg_dp)
    res_nobg_dp_pil = Image.fromarray(res_nobg_dp)

    return res_bg_dp_pil, res_nobg_dp_pil

if __name__ == "__main__":
    main()
