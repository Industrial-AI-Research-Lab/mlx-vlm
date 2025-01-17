import argparse
from .utils import convert

def convert_external_adapter(model, adapter_source, adapter_path):

    raise NotImplementedError("Convert external adapter is not implemented")

def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert external adapter")
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True, 
        help="Path to the adapter"
    )
    parser.add_argument(
        "--adapter_source",
        type=str,
        required=True,
        help="Framework that was used in model production",
        default="Unsloth"
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameters, ignored if -q is given.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="converted_adapter",
        help="Path to save the trained adapter",
    )
    args = parser.parse_args()
    convert_external_adapter(args.model_path, args.adapter_name, args.adapter_path)

def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))

if __name__ == "__main__":
    main()