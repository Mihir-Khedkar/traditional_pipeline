import json
import os

config = {}

config["input_path"] = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\inputs'
config["output_path"] = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\outputs'

filename = r'C:\Users\Mihir Khedkar\Desktop\traditional_pipeline\config.json'

with open(filename, 'w') as file:
    json.dump(config, file, indent=4)