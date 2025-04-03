import torch

class DebugContext:
    _enabled = False  # 控制 Hook 是否生效
    _i = 0
    _prefix = ''
    _ctrl_type = 'canny'
    _scale = '0.8'

    @classmethod
    def set_prefix(cls, prefix='', ctrl_type='canny', scale='0.8'):
        cls._prefix = prefix
        cls._ctrl_type = ctrl_type
        cls._scale = scale

    @classmethod
    def set_i(cls, i):
        cls._i = i

    @classmethod
    def set_attn_index(cls, index):
        cls._index = index

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def is_enabled(cls):
        return cls._enabled

    @classmethod
    def save(cls, dir, value, info='', enable_i=True):
        if cls._enabled:
            info_str = f'_{info}' if info != '' else ''
            i_str = f'_{cls._i}' if enable_i else ''
            torch.save(value, f'{dir}/{cls._prefix}_{cls._ctrl_type}_{cls._scale}{info_str}{i_str}.pt')

# Hook 记录函数
def debug_hook(dir, value, info='', enable_i=True):
    DebugContext.save(dir, value, info, enable_i)