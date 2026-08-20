import argparse
import os
import sys

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Validate dependencies
try:
    import cv2
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install opencv-python tensorflow numpy")
    sys.exit(1)

print("Script running...")

parser = argparse.ArgumentParser(
    description="Compress an image using a trained model.",
    epilog="""
Examples:
  python test_compression.py --input photo.jpg --output compressed.jpg
  python test_compression.py --input C:\\path\\to\\image.png --output result.png --model my_model.h5
  python test_compression.py --demo (uses first image from dataset/train folder)
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--input", help="Path to input image file (e.g. photo.jpg, C:\\path\\to\\image.png)")
parser.add_argument("--output", help="Path to save compressed image (e.g. compressed.jpg)")
parser.add_argument("--model", default="image_compression_model.h5", help="Path to trained model file (default: image_compression_model.h5)")
parser.add_argument("--demo", action="store_true", help="Run demo: automatically find and compress first image from dataset/train")
parser.add_argument("--guide", action="store_true", help="Print a step-by-step usage guide and exit")

# If no arguments provided, suggest --demo
if len(sys.argv) == 1:
    print("No arguments provided. Try running with --demo to test with an image from your dataset:")
    print("  python test_compression.py --demo\n")
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()

def print_guide():
    print("""
Quick step-by-step guide

1) Install prerequisites (run once):
   pip install opencv-python tensorflow numpy

2) Ensure your trained model file is reachable:
   - Default name the script expects: image_compression_model.h5
   - Place it in the same folder as the script, or pass --model "C:\\full\\path\\model.h5"

3) Test quickly with demo mode (uses first image under dataset/train if present):
   python test_compression.py --demo
   Output: demo_compressed.jpg

4) Compress a specific image:
   python test_compression.py --input "C:\\path\\to\\photo.jpg" --output "C:\\path\\to\\out.jpg"
   (Add --model if your model has a different name)

5) Troubleshooting:
   - "Model file not found": verify path and filename, and model is a Keras .h5 compatible with your TF version.
   - "Could not read image": verify the input path and that the file is a valid image.
   - Permission errors when writing output: check output directory exists and is writable.

6) Validation mode (no compression) can be used to check environment:
   python test_compression.py --check

If you want, run: python test_compression.py --guide
""")
    sys.exit(0)

# If --guide requested, show instructions and exit
if "--guide" in sys.argv:
    # args not yet parsed in some flows; print guide immediately
    print_guide()

# Handle demo mode
if args.demo:
    dataset_train = os.path.join(os.path.dirname(__file__), "dataset", "train")
    if os.path.isdir(dataset_train):
        images = [f for f in os.listdir(dataset_train) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            args.input = os.path.join(dataset_train, images[0])
            args.output = "demo_compressed.jpg"
            print(f"Demo mode: using {args.input}")
        else:
            print(f"Error: No images found in {dataset_train}")
            sys.exit(1)
    else:
        print(f"Error: Dataset folder not found: {dataset_train}")
        sys.exit(1)

# Validate required arguments
if not args.input or not args.output:
    parser.print_help()
    print("\nError: --input and --output are required (or use --demo)")
    sys.exit(1)

# Validate input file exists
if not os.path.isfile(args.input):
    print(f"Error: Input file not found: {args.input}")
    sys.exit(1)

# Validate model file exists
if not os.path.isfile(args.model):
    print(f"Error: Model file not found: {args.model}")
    print(f"Expected: {os.path.abspath(args.model)}")
    sys.exit(1)

try:
    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

try:
    print(f"Reading image from: {args.input}")
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not read image (unsupported format or corrupted file): {args.input}")
        sys.exit(1)
    print(f"Image shape: {img.shape}")

    print("Resizing to 128x128...")
    img_resized = cv2.resize(img, (128, 128))
    img_norm = img_resized / 255.0

    print("Running compression...")
    compressed = model.predict(np.expand_dims(img_norm, axis=0))

    print("Saving compressed image...")
    output_img = (compressed[0] * 255).astype("uint8")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, output_img)
    print(f"Saved: {os.path.abspath(args.output)}")

except Exception as e:
    print(f"Error during compression: {e}")
    sys.exit(1)

print("Compression complete!")


