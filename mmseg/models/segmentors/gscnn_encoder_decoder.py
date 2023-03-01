import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mmcv
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
import warnings


@SEGMENTORS.register_module()
class GSCNNEncoderDecoder(EncoderDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)
        
    def forward_train(self, img, img_metas, gt_semantic_seg, img_edge, gt_edge):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            img_edge (Tensor): Input image edges.
            gt_edge (Tensor): Ground truth edges.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg, img_edge, gt_edge)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def forward_test(self, imgs, img_metas, img_edge, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_edge, 'img_edges'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], img_edge[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, img_edge, **kwargs)

    def simple_test(self, img, img_meta, img_edge, rescale=True):
        """Simple test with single image."""
        seg_logit, edge_logit = self.inference(img, img_meta, rescale, img_edge)

        # b.w.
        n_ch = seg_logit.size(1)
        if n_ch > 1:
            seg_pred = seg_logit.argmax(dim=1)
        else:
            seg_pred = (seg_logit > 0.5).byte().squeeze(1)
        edge_pred = edge_logit.sigmoid()

        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            edge_pred = edge_pred.unsqueeze(0)
            return seg_pred, edge_pred
        seg_pred = seg_pred.cpu().numpy()
        edge_pred = edge_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        edge_pred = list(edge_pred)
        return seg_pred, edge_pred

    def aug_test(self, imgs, img_metas, img_edges, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit, edge_logit = self.inference(imgs[0], img_metas[0], img_edges[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit, cur_edge_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
            edge_logit += cur_edge_logit
        seg_logit /= len(imgs)
        edge_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        edge_pred = edge_logit.sigmoid()
        seg_pred = seg_pred.cpu().numpy()
        edge_pred = edge_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        edge_pred = list(edge_pred)
        return seg_pred, edge_pred

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, img_edge, gt_edge):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     img_edge, gt_edge,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas, img_edge):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits, edge_logits = self.decode_head.forward_test(x, img_metas, img_edge, self.test_cfg)
        return seg_logits, edge_logits

    def encode_decode(self, img, img_metas, img_edge):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        seg_out, edge_out = self._decode_head_forward_test(x, img_metas, img_edge)
        seg_out = resize(
            input=seg_out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        edge_out = resize(
            input=edge_out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_out, edge_out

    def inference(self, img, img_meta, rescale, img_edge):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit, edge_logit = self.slide_inference(img, img_meta, rescale, img_edge)
        else:
            seg_logit, edge_logit = self.whole_inference(img, img_meta, rescale, img_edge)

        # b.w.
        n_ch = seg_logit.size(1)
        if n_ch > 1:
            seg_out = F.softmax(seg_logit, dim=1)
        else:
            seg_out = F.sigmoid(seg_logit)
        edge_out = F.sigmoid(edge_logit)

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                seg_out = seg_out.flip(dims=(3, ))
                edge_out = edge_out.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                seg_out = seg_out.flip(dims=(2, ))
                edge_out = edge_out.flip(dims=(2, ))

        return seg_out, edge_out

    def slide_inference(self, img, img_meta, rescale, img_edge):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        seg_preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        edge_preds = img_edge.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_img_edge = img_edge[:, :, y1:y2, x1:x2]
                crop_seg_logit, crop_edge_logit = self.encode_decode(crop_img, img_meta, crop_img_edge)
                seg_preds += F.pad(crop_seg_logit, (int(x1), int(seg_preds.shape[3] - x2), int(y1), int(seg_preds.shape[2] - y2)))
                edge_preds += F.pad(crop_edge_logit, (int(x1), int(edge_preds.shape[3] - x2), int(y1), int(edge_preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        seg_preds = seg_preds / count_mat
        edge_preds = edge_preds / count_mat
        if rescale:
            seg_preds = resize(
                seg_preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            edge_preds = resize(
                edge_preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_preds, edge_preds

    def whole_inference(self, img, img_meta, rescale, img_edge):
        """Inference with full image."""

        seg_logit, edge_logit = self.encode_decode(img, img_meta, img_edge)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            edge_logit = resize(
                edge_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit, edge_logit

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        if palette.shape[0] == 1:
            palette = np.insert(palette, 0, (0, 0, 0), 0)

        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img