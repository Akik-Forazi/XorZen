import torch
import os
import argparse

def dump_layer(module, input_data, output_dir, layer_name):
    """
    Hooks into a module, runs the input, and saves IO as raw binary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run forward pass
    output = module(input_data)
    
    # Save IO
    input_data.detach().numpy().tofile(os.path.join(output_dir, f"{layer_name}_x.bin"))
    output.detach().numpy().tofile(os.path.join(output_dir, f"{layer_name}_y.bin"))
    
    # Save weights if it's a parameter-bearing layer
    if hasattr(module, 'weight'):
        module.weight.detach().numpy().tofile(os.path.join(output_dir, f"{layer_name}_weight.bin"))
        
    print(f"Golden vectors for {layer_name} saved to {output_dir}")

# Example Usage setup for the user
if __name__ == "__main__":
    # This is a template for the user to hook their specific layer
    print("This script is a utility to dump golden IO tensors.")
    print("Usage: Import this script and call dump_layer(my_module, my_input, 'tests/parity', 'my_layer_name')")
