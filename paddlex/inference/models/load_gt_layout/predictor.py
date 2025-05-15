# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any, List

from ...common.batch_sampler import ImageBatchSampler
from ..base import BasePredictor
from ..object_detection.result import DetResult

global IMG_PATHS
IMG_PATHS = []


class GTDetPredictor(BasePredictor):

    entities = "PP-DocLayout_plus-L"

    _FUNC_MAP = {}

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_dir="/root/.paddlex/official_models/PP-DocLayout_plus-L/"
        )
        label_dir = "/paddle/project/github/LabelLayout/datasets/label_studio/"
        self.labels_dir = f"{label_dir}/labels"
        notes_path = f"{label_dir}/notes.json"
        with open(notes_path, "r") as f:
            notes = json.load(f)

        self.categories_map = {}
        for categories in notes["categories"]:
            id = int(categories["id"])
            name = categories["name"]
            self.categories_map[id] = name

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return DetResult

    def process(self, batch_data: List[Any], *args, **kwargs):
        all_boxes = []
        for img, fn in zip(batch_data.instances, IMG_PATHS):
            gt_file = os.path.basename(fn)[:-4] + ".txt"
            gt_path = os.path.join(self.labels_dir, gt_file)
            with open(gt_path, "r") as f:
                lines = f.readlines()
            boxes = []
            for line in lines:
                line = line.strip().split(" ")
                category_id = int(line[0])
                label = self.categories_map[category_id]
                img_h, img_w = img.shape[:2]
                center_x = float(line[1]) * img_w
                center_y = float(line[2]) * img_h
                w = float(line[3]) * img_w
                h = float(line[4]) * img_h
                x0 = center_x - w / 2
                y0 = center_y - h / 2
                x1 = center_x + w / 2
                y1 = center_y + h / 2
                box = [x0, y0, x1, y1]
                boxes.append(
                    {
                        "cls_id": category_id,
                        "label": label,
                        "coordinate": box,
                        "score": 1.0,
                    }
                )
            all_boxes.append(boxes)

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_data.instances,
            "boxes": all_boxes,
        }
