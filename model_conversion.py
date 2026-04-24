import os
import time
import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


def convert_saved_model_to_tflite(saved_model_dir, tflite_path, quantize=True, representative_samples=None):
    """Convert a TensorFlow SavedModel to TFLite with optional int8 quantization."""
    if tf is None:
        raise ImportError("TensorFlow is required for model conversion. Install tensorflow or tensorflow-cpu.")

    if not os.path.exists(saved_model_dir):
        raise FileNotFoundError(f"SavedModel path not found: {saved_model_dir}")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_samples is not None:
            converter.representative_dataset = representative_samples
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    return tflite_path


def load_tflite_interpreter(model_path):
    """Load a TFLite interpreter and return it ready for inference."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required to load the TFLite interpreter.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_embedding_input(face_img, target_size=(160, 160)):
    """Prepare a face image for a TFLite embedding model."""
    img = face_img.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def infer_embedding(interpreter, face_img):
    """Run inference on a TFLite model to obtain a 128-D embedding."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    processed = preprocess_embedding_input(face_img, tuple(input_details[0]["shape"][1:3]))
    if input_details[0]["dtype"] == np.uint8:
        processed = (processed * 255.0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], processed)
    start = time.perf_counter()
    interpreter.invoke()
    end = time.perf_counter()

    embedding = interpreter.get_tensor(output_details[0]["index"]) 
    return embedding.flatten(), (end - start)


def benchmark_tflite(interpreter, face_img, run_count=20):
    """Measure average inference time for the interpreter."""
    _, _ = infer_embedding(interpreter, face_img)
    times = []
    for _ in range(run_count):
        _, elapsed = infer_embedding(interpreter, face_img)
        times.append(elapsed)
    return float(np.mean(times)), float(np.std(times))


def build_conversion_report(saved_model_dir, tflite_path, input_face, runs=20):
    """Return a report comparing model sizes and TFLite inference latency."""
    if not tf:
        raise ImportError("TensorFlow is required for conversion reporting.")

    if not os.path.exists(saved_model_dir):
        raise FileNotFoundError(f"SavedModel path not found: {saved_model_dir}")

    saved_size = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(saved_model_dir) for file in files)
    tflite_size = os.path.getsize(tflite_path)

    interpreter = load_tflite_interpreter(tflite_path)
    avg_latency, std_latency = benchmark_tflite(interpreter, input_face, run_count=runs)

    return {
        "saved_model_size_bytes": saved_size,
        "tflite_size_bytes": tflite_size,
        "compression_ratio": saved_size / tflite_size if tflite_size > 0 else None,
        "average_latency_sec": avg_latency,
        "latency_std_sec": std_latency,
        "tflite_path": tflite_path
    }


if __name__ == '__main__':
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description='Convert TensorFlow model to TFLite and benchmark it.')
    parser.add_argument('--saved_model', required=True, help='Path to the SavedModel directory')
    parser.add_argument('--tflite_path', required=True, help='Destination .tflite file path')
    parser.add_argument('--sample_image', required=True, help='Sample face image for latency benchmark')
    parser.add_argument('--runs', type=int, default=20, help='Number of inference runs for benchmarking')
    args = parser.parse_args()

    print('Converting model...')
    convert_saved_model_to_tflite(args.saved_model, args.tflite_path, quantize=True)
    print('Loading sample image...')
    sample = cv2.imread(args.sample_image)
    if sample is None:
        raise FileNotFoundError(f'Sample image not found: {args.sample_image}')

    report = build_conversion_report(args.saved_model, args.tflite_path, sample, runs=args.runs)
    print('Conversion report:')
    print(report)
