import torch


class SegmentTensorManager:
    def __init__(self, seg_mode, segments, segment_size):
        self.seg_mode = seg_mode
        self.segments = segments
        self.segment_size = segment_size

class InputSegmentTensorManager(SegmentTensorManager):
    def __init__(self, input_tensor, seg_mode, segments, segment_size):
        super().__init__(seg_mode=seg_mode, segments=segments, segment_size=segment_size)
        self.input_tensor = input_tensor

    def iter_view(self):
        if self.seg_mode == "input":
            new_shape = list(self.input_tensor.shape[:-1]) + [self.segments, self.segment_size]
            ndim = self.input_tensor.ndim + 1
            permute_order_tuple = (ndim - 2, *range(ndim - 2), ndim - 1)
            return self.input_tensor.view(new_shape).permute(permute_order_tuple) # (segments, ..., segment_size)
        elif self.seg_mode == 'weight':
            expand_args = [self.segments] + [-1] * (self.input_tensor.ndim)
            return self.input_tensor.unsqueeze(0).expand(expand_args) # (segments (repeated), ..., in)

    def total_view(self):
        return self.input_tensor.unsqueeze(0)# (1, ..., in)

class WeightSegmentTensorManager(SegmentTensorManager):
    def __init__(self, weight_tensor, seg_mode, segments, segment_size):
        super().__init__(seg_mode=seg_mode, segments=segments, segment_size=segment_size)
        assert weight_tensor.ndim == 2, "weight tensor shape failed."

        self.out_features, self.in_features = weight_tensor.shape
        self.weight_tensor = weight_tensor # (out, in)

        self.packed = False

    def device(self):
        return self.weight_tensor.device

    def to(self, device):
        self.weight_tensor = self.weight_tensor.to(device)
        return self

    def iter_view(self):
        if self.packed:
            return self.weight_tensor
        if self.seg_mode == 'input':
            return self.weight_tensor.view(self.out_features, self.segments, self.segment_size).permute(1, 0, 2) # (segments, out, segment_size)
        elif self.seg_mode == 'weight':
            return self.weight_tensor.view(self.segments, self.segment_size, self.in_features) # (segments, segment_size, in)

    def total_view(self):
        return self.weight_tensor.unsqueeze(0) # (1, out, in)

    def replace_with_segments_layout(self, segmented_weights, packed=False):
        if packed:
            # for int4, only store as iter-view
            # because we only need to forward when inferring
            # normal shape input-seg: (segments, out, segment_size)
            # normal shape weight-seg: (segments, segment_size, in)
            # packed shape input-seg: (segments, out * segment_size // 2) dtype=uint8
            # packed shape weight-seg: (segments, segment_size * in // 2) dtype=uint8
            self.weight_tensor = segmented_weights
            self.packed = packed
            return

        if self.seg_mode == 'input':
            # (segments, out, segment_size)
            segmented_weights = segmented_weights.permute(1, 0, 2).contiguous()
            self.weight_tensor = segmented_weights.view(
                self.out_features, 
                self.in_features
            )
        elif self.seg_mode == 'weight':
            # (segments, segment_size, in)
            self.weight_tensor = segmented_weights.view(
                self.out_features,
                self.in_features
            )


def segmented_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (segments, ..., in)
    b: (segments, out, in)
    -> (segments, ..., out)
    """
    assert a.shape[0] == b.shape[0], "segments must match"

    b_t = b.transpose(-1, -2)  # (segments, in, out)
    while b_t.ndim < a.ndim:
        b_t = b_t.unsqueeze(1)
    return a @ b_t
