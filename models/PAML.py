import torch
import torch.nn.functional as F
from torch import nn
import os
from util import box_ops
from util.misc import (NestedTensor, get_world_size, is_dist_avail_and_initialized)
from .albef.VL_Transformer_ITM import VL_Transformer_ITM
from .backbone import build_backbone
from torchvision.transforms import Resize
from .transformer import build_visual_encoder
from .decoder import build_vg_decoder
from pytorch_pretrained_bert.modeling import BertModel
def keyframe_extraction(x, target_len=40):
  """
  使用均匀选择方法抽取关键帧。

  Args:
    x: 输入张量，形状为 (batch_size, seq_len, feature_size)
    target_len: 目标序列长度

  Returns:
    抽取关键帧后的张量，形状为 (batch_size, target_len, feature_size)
  """
  batch_size, seq_len, feature_size = x.size()
  indices = torch.linspace(0, seq_len - 1, target_len).long() # 生成均匀间隔的索引
  x_extracted = x[:, indices, :]
  return x_extracted
def interpolate_tensor(x, target_len=400):
  """
  使用线性插值扩展张量到目标长度。

  Args:
    x: 输入张量，形状为 (batch_size, seq_len, feature_size)
    target_len: 目标序列长度

  Returns:
    插值后的张量，形状为 (batch_size, target_len, feature_size)
  """
  x = x.transpose(1, 2)  # 交换维度，使 token 维度成为最后一个维度
  x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
  x = x.transpose(1, 2)  # 恢复原始维度顺序
  return x

class PAML(nn.Module):
    def __init__(self, pretrained_weights, args=None):
        """ Initializes the model."""
        super().__init__()

        # Image feature encoder (CNN + Transformer encoder)
        self.backbone = build_backbone(args)
        self.trans_encoder = build_visual_encoder(args)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.trans_encoder.d_model, kernel_size=1)

        self.word_p1 = nn.Parameter(torch.tensor([0.6], dtype=torch.float32), requires_grad=True)
        self.word_p2 = nn.Parameter(torch.tensor([0.3], dtype=torch.float32), requires_grad=True)
        self.word_p3 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32), requires_grad=True)

        # Text feature encoder (BERT)(语言模型 easy)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_proj = nn.Linear(args.bert_output_dim, args.hidden_dim)
        self.bert_output_layers = args.bert_output_layers
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

        # visual grounding
        self.trans_decoder = build_vg_decoder(args)

        hidden_dim = self.trans_encoder.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.vl_encoder = VL_Transformer_ITM(text_encoder='./checkpoints/bert-base-uncased',
                                   config_bert=os.path.join(os.path.dirname(os.path.abspath(__file__)),  # VLP预训练模型
                                                            'albef/config_bert.json'))
        checkpoint_path = os.path.join('./checkpoints', 'ALBEF.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        checkpoint = {k.replace('.bert', ''): v for k, v in checkpoint.items()}

        msg = self.vl_encoder.load_state_dict(checkpoint, strict=False)

        self.vl_encoder.to('cuda')
        self.visu_proj = nn.Linear(768,256)
        self.text_proj = nn.Linear(768,256)

        # if pretrained_weights:
        #     self.load_pretrained_weights(pretrained_weights)

    def get_masks(self, images_mask):
        torch_resize = Resize([20, 20])
        visu_masks = torch_resize(images_mask)
        visu_masks = visu_masks.to(torch.bool)
        visu_masks = visu_masks.flatten(1)

        return visu_masks

    def load_pretrained_weights(self, weights_path):
        def load_weights(module, prefix, weights):
            module_keys = module.state_dict().keys()
            weights_keys = [k for k in weights.keys() if prefix in k]
            update_weights = dict()
            for k in module_keys:
                prefix_k = prefix+'.'+k
                if prefix_k in weights_keys:
                    update_weights[k] = weights[prefix_k]
                else:
                    print(f"Weights of {k} are not pre-loaded.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['model']
        load_weights(self.backbone, prefix='backbone', weights=weights)
        load_weights(self.trans_encoder, prefix='transformer', weights=weights)
        load_weights(self.input_proj, prefix='input_proj', weights=weights)


    def forward(self, image, image_mask, word_id, word_mask, feature):

        N = image.size(0)
        word_mask_t = word_mask
        word_mask = word_mask + 0
        img_feat,word_feat = self.vl_encoder(image,word_id,word_mask)
        word_mask = ~word_mask_t                 #word_feat(B,40,256)
        mask = self.get_masks(image_mask)
        img_feat = self.visu_proj(img_feat)
        img_feat = interpolate_tensor(img_feat)
        word_feat = self.text_proj(word_feat)
        img_feat = img_feat.permute(1,0,2)
        img_feat, mask, pos_embed = self.trans_encoder(img_feat, mask,0)
        word_feat = word_feat.permute(1,0,2)
        #获得文本信息以及图像信息之后开始猛操作
        # Discriminative feature encoding + Multi-stage reasoning
        hs = self.trans_decoder(img_feat, mask, 0, word_feat, word_mask, 0)

        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_boxes': outputs_coord[-1]}
        #最新的一个query是作为预测BBOX，其他的中间query是作为辅助输出
        if self.training:
            out['aux_outputs'] = [{'pred_boxes': b} for b in outputs_coord[:-1]]
        return out




class VGCriterion(nn.Module):
    """ This class computes the loss for VLTVG."""
    def __init__(self, weight_dict, loss_loc, box_xyxy):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict

        self.box_xyxy = box_xyxy

        self.loss_map = {'loss_boxes': self.loss_boxes}

        self.loss_loc = self.loss_map[loss_loc]

    def loss_boxes(self, outputs, target_boxes, num_pos):
        """Compute the losses related to the bounding boxes (the L1 regression loss and the GIoU loss)"""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'] # [B, #query, 4]
        target_boxes = target_boxes[:, None].expand_as(src_boxes)

        src_boxes = src_boxes.reshape(-1, 4) # [B*#query, 4]
        target_boxes = target_boxes.reshape(-1, 4) #[B*#query, 4]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        if not self.box_xyxy:
            src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - box_ops.box_pair_giou(src_boxes, target_boxes)
        losses['giou'] = (loss_giou[:, None]).sum() / num_pos
        return losses


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """
        gt_boxes = targets['bbox']
        pred_boxes = outputs['pred_boxes']

        losses = {}
        B, Q, _ = pred_boxes.shape
        num_pos = avg_across_gpus(pred_boxes.new_tensor(B*Q))
        loss = self.loss_loc(outputs, gt_boxes, num_pos)
        losses.update(loss)

        # Apply the loss function to the outputs from all the stages
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_loc(aux_outputs, gt_boxes, num_pos)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format we expect"""
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        """ Perform the computation"""
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes']

        assert len(boxes) == len(rsz_sizes)
        assert rsz_sizes.shape[1] == 2

        boxes = boxes.squeeze(1)

        # Convert to absolute coordinates in the original image
        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1)
        boxes = boxes.clamp(min=0)
        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1)
        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1))

        return boxes


def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




def build_vgmodel(args):
    device = torch.device(args.device)

    model = PAML(pretrained_weights=args.load_weights_path, args=args)

    weight_dict = {'loss_cls': 1, 'l1': args.bbox_loss_coef}
    weight_dict['giou'] = args.giou_loss_coef
    weight_dict.update(args.other_loss_coefs)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = VGCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc, box_xyxy=args.box_xyxy)
    criterion.to(device)

    postprocessor = PostProcess(args.box_xyxy)

    return model, criterion, postprocessor
