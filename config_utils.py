import json
import base64
import os
from typing import Optional

# Default configuration path
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

def get_model_name(config_path: Optional[str] = None) -> str:
    
    config_path = config_path or CONFIG_PATH
    
   
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config.get('model_config', {})
    use_encoded = model_config.get('use_encoded', False)
    
    if use_encoded:
        encoded_name = model_config.get('encoded_name', '')
        if encoded_name:
            return base64.b64decode(encoded_name).decode('utf-8')
        

# Utility function to encode a model name for configuration
def encode_model_name(model_name: str) -> str:
    """
    Encode a model name to base64 for storing in configuration.
    
    Args:
        model_name: The model name to encode
        
    Returns:
        The base64 encoded model name
    """
    return base64.b64encode(model_name.encode('utf-8')).decode('utf-8')

# Example usage
