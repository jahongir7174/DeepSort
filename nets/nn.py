import math

import cv2
import numpy
import torch
from torchvision import transforms

from utils import util


def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
        if self.training:
            return x
        self.anchors, self.strides = (x.transpose(0, 1) for x in util.make_anchors(x,
                                                                                   self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class Block(torch.nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.add = s != 1 or in_ch != out_ch
        self.relu = torch.nn.ReLU(inplace=False)

        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_ch, out_ch, 3, s, 1, bias=False),
                                         torch.nn.BatchNorm2d(out_ch, 0.001, 0.03),
                                         torch.nn.ReLU(inplace=False))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                                         torch.nn.BatchNorm2d(out_ch, 0.001, 0.03))

        if self.add:
            self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(in_ch, out_ch, 1, s, 0, bias=False),
                                             torch.nn.BatchNorm2d(out_ch, 0.001, 0.03))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.add:
            x = self.conv3(x)

        return self.relu(x + y)


class ReID(torch.nn.Module):
    def __init__(self):
        super().__init__()
        filters = [3, 64, 128, 256, 512]
        p1 = [torch.nn.Conv2d(filters[0], filters[1], 3, 1, 1),
              torch.nn.BatchNorm2d(filters[1], 0.001, 0.03),
              torch.nn.ReLU(inplace=False),
              torch.nn.MaxPool2d(3, 2, padding=1)]
        p2 = [Block(filters[1], filters[1], 1),
              Block(filters[1], filters[1], 1)]
        p3 = [Block(filters[1], filters[2], 2),
              Block(filters[2], filters[2], 1)]
        p4 = [Block(filters[2], filters[3], 2),
              Block(filters[3], filters[3], 1)]
        p5 = [Block(filters[3], filters[4], 2),
              Block(filters[4], filters[4], 1)]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

        self.pool = torch.nn.AvgPool2d((8, 4), 1)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x.div(x.norm(p=2, dim=1, keepdim=True))


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.reid = torch.load(model_path,
                               map_location=torch.device(self.device))['model']

        self.reid.to(self.device)
        if torch.cuda.is_available():
            self.reid.half()
        self.size = (64, 128)
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(numpy.float32) / 255., size)

        batch = [self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops]
        return torch.cat(batch, dim=0).float()

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            if torch.cuda.is_available():
                im_batch = im_batch.half()
            features = self.reid(im_batch)
        return features.cpu().numpy()


class DeepSort(object):
    def __init__(self,
                 model_path='weights/reid.pt',
                 max_dist=0.2, min_confidence=0.3,
                 nms_max_overlap=0.5, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        metric = util.NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = util.Tracker(metric,
                                    max_iou_distance=max_iou_distance,
                                    max_age=max_age, n_init=n_init)
        self.height, self.width = 1, 1

    def update(self, bbox_xywh, confidences, oids, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tl_wh = self._xywh_to_tlwh(bbox_xywh)
        detections = [util.Detection(bbox_tl_wh[i], conf, features[i], oid) for i, (conf, oid) in
                      enumerate(zip(confidences, oids)) if conf > self.min_confidence]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            track_oid = track.oid
            outputs.append(numpy.array([x1, y1, x2, y2, track_id, track_oid], dtype=numpy.int))
        if len(outputs) > 0:
            outputs = numpy.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(box_xy_wh):
        box_tl_wh = box_xy_wh.copy()
        box_tl_wh[:, 0] = box_xy_wh[:, 0] - box_xy_wh[:, 2] / 2.
        box_tl_wh[:, 1] = box_xy_wh[:, 1] - box_xy_wh[:, 3] / 2.
        return box_tl_wh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):

        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = numpy.array([])
        return features
