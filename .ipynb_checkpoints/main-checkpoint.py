import argparse
import importlib.util

def load_config_py(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Python config file")
    args = parser.parse_args()

    config = load_config_py(args.config)

    from trainers.trainer import train  
    train(config)
