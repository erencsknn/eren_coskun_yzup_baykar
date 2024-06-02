from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class UDD_VDD_Dataset(BaseSegDataset):
    METAINFO = dict(
    classes=('vegetation', 'wall', 'road', 'vehicle', 'potential landing area', 'other', 'water'),
        palette=[[107, 142, 35], [102, 102, 156], [128, 64, 128], [0, 0, 142],
                 [70, 70, 70], [0, 0, 0], [0, 200, 200]])
    

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.jpg',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)