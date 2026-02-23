import yaml
from segquant.config import Calibrate, DType, Optimum, SegPattern

def parse_enum(value, enum_cls):
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)

def convert_config(config: dict):
    config = config.copy()

    if "search_patterns" in config:
        config["search_patterns"] = [
            parse_enum(p, SegPattern) for p in config["search_patterns"]
        ]

    if "opt" in config and "type" in config["opt"]:
        config["opt"]["type"] = parse_enum(config["opt"]["type"], Optimum)

    if "calib" in config and "type" in config["calib"]:
        config["calib"]["type"] = parse_enum(config["calib"]["type"], Calibrate)

    if "input_quant" in config and "type" in config["input_quant"]:
        config["input_quant"]["type"] = parse_enum(config["input_quant"]["type"], DType)

    if "weight_quant" in config and "type" in config["weight_quant"]:
        config["weight_quant"]["type"] = parse_enum(config["weight_quant"]["type"], DType)

    return config

def parse_yaml(model_type, layer_type, quant_type, file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    this_configs = {}
    for name, config in data.items():
        parsed_config = {}
        for k, v in config.items():
            parsed_config[k] = convert_config(v)
        this_configs[f'{model_type}-{layer_type}-{quant_type}-{name}'] = parsed_config

    return this_configs

if __name__ == "__main__":
    model_type = 'flux'
    quant_type = 'int-w8a8'
    res = parse_yaml(model_type, quant_type, f'config/{model_type}/{quant_type}.yaml')
    print(res)