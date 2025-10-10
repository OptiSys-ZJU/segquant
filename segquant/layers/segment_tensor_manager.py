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
            expand_args = [self.segments] + [-1] * (self.input_tensor.ndim - 1)
            return self.input_tensor.unsqueeze(0).expand(expand_args) # (segments (repeated), ..., in)

    def total_view(self):
        return self.input_tensor.unsqueeze(0)# (1, ..., in)

class WeightSegmentTensorManager(SegmentTensorManager):
    def __init__(self, weight_tensor, seg_mode, segments, segment_size):
        super().__init__(seg_mode=seg_mode, segments=segments, segment_size=segment_size)
        assert weight_tensor.dims == 2, 'weight tensor shape failed.'

        self.out_features, self.in_features = weight_tensor.shape
        self.weight_tensor = weight_tensor # (out, in)
    
    def iter_view(self):
        if self.seg_mode == 'input':
            return self.weight_tensor.view(self.out_features, self.segments, self.segment_size).permute(1, 0, 2) # (segments, out, segment_size)
        elif self.seg_mode == 'weight':
            return self.weight_tensor.view(self.segments, self.segment_size, self.in_features) # (segments, segment_size, in)
    
    def total_view(self):
        return self.weight_tensor.unsqueeze(0) # (1, out, in)
