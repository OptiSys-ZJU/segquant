from benchmark.config import AffineConfig
import os
import json

config = AffineConfig.to_dict()
config["stepper"]["max_timestep"] = 20

if os.path.exists("affiner_config.json"):
    with open("affiner_config.json", "r") as f:
        try:
            configs = json.load(f)
        except json.JSONDecodeError:
            configs = []

        if config in configs:
            print("Config already exists")
            pass
        else:
            print("Config does not exist, appending config")
            configs.append(config)
            with open("affiner_config.json", "w") as f2:
                json.dump(configs, f2, indent=4)
else:
    configs = []
    with open("affiner_config.json", "w") as f:
        configs.append(config)
        json.dump(configs, f, indent=4)



