import torch
from ultralytics.utils.loss import v8DetectionLoss


class PcbSmallObjLoss(v8DetectionLoss):
    def __init__(self, model, small_weight=3.0, medium_weight=1.5,
                 small_thresh=16, medium_thresh=32, dfl_weight=2.0,
                 warmup_epochs=50):
        super().__init__(model)
        self.small_weight = small_weight
        self.medium_weight = medium_weight
        self.small_thresh = small_thresh ** 2
        self.medium_thresh = medium_thresh ** 2
        self.dfl_weight = dfl_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self._init_done = True
        print("=" * 60)
        print("✅ PcbSmallObjLossEnhanced 初始化成功!")
        print(f"📊 小目标权重: {small_weight}, 中目标权重: {medium_weight}")
        print(f"📏 小目标阈值: {small_thresh}px, 中目标阈值: {medium_thresh}px")
        print(f"🎯 DFL权重: {dfl_weight}, 预热轮次: {warmup_epochs}")
        print("=" * 60)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __call__(self, preds, batch):
        loss, loss_items = super().__call__(preds, batch)

        # 1. warmup 渐进
        if self.current_epoch < self.warmup_epochs:
            p = self.current_epoch / self.warmup_epochs
            small_w = 1 + (self.small_weight - 1) * p
            medium_w = 1 + (self.medium_weight - 1) * p
            dfl_w = 1 + (self.dfl_weight - 1) * p
        else:
            small_w, medium_w, dfl_w = self.small_weight, self.medium_weight, self.dfl_weight

        bboxes = batch['bboxes']
        if bboxes.shape[0] == 0:
            return loss, loss_items

        areas = bboxes[:, 2] * bboxes[:, 3] * (640 ** 2)
        small_mask = areas < self.small_thresh
        medium_mask = (areas >= self.small_thresh) & (areas < self.medium_thresh)

        # 2. 面积细分 + 圆形度
        w_area = torch.ones_like(areas)
        w_area = torch.where(areas < 64, 4.0, w_area)
        w_area = torch.where((areas >= 64) & (areas < 256), 2.5, w_area)
        peri = 2 * (bboxes[:, 2] + bboxes[:, 3]) * 640
        circularity = 4 * 3.1416 * areas / (peri ** 2 + 1e-7)
        circ_w = 1 + (circularity - 0.7).clamp_min(0) * 2
        w = w_area * circ_w

        # 3. 焊盘掩膜
        if 'pad_mask' in batch:
            from ultralytics.utils.ops import xywhn2xyxy
            ctr = xywhn2xyxy(bboxes, w=640, h=640)[:, :2].long()
            inside = batch['pad_mask'][batch['batch_idx'], 0, ctr[:, 1], ctr[:, 0]]
            w = w * (inside * 0.5 + 1.0)

        # 4. 困难样本
        if hasattr(self, 'iou'):
            iou = self.iou.max(dim=0)[0]
            hard = small_mask & (iou < 0.5)
            w = w * (hard * 1.0 + 1.0)

        loss_items[0] *= w
        loss_items[1] *= w
        loss_items[2] *= w
        loss_items[3] *= w

        # 5. 层衰减
        for i, _ in enumerate(preds):
            if i > 0:
                loss_items[3][..., i] *= 0.6

        # 原中小目标加权
        if small_mask.any():
            loss_items[0][small_mask] *= small_w / w[small_mask]
            loss_items[1][small_mask] *= small_w / w[small_mask]
            loss_items[3][small_mask] *= small_w / w[small_mask]
            loss_items[2][small_mask] *= dfl_w
        if medium_mask.any():
            loss_items[0][medium_mask] *= medium_w
            loss_items[1][medium_mask] *= medium_w
            loss_items[3][medium_mask] *= medium_w

        return loss_items.sum(), loss_items