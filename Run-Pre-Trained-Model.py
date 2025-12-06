import argparse
import cv2
import supervision as sv
import torch
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run RF-DETR model on webcam')
parser.add_argument(
    '--model',
    type=str,
    default='base',
    choices=['nano', 'small', 'base', 'medium', 'large'],
    help='Model version to use: nano (fastest), small, base, medium, or large (most accurate). Default: base'
)
parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='Path to custom checkpoint weights file (e.g., checkpoint_best_total.pth). If not provided, uses pretrained weights.'
)
parser.add_argument(
    '--num-classes',
    type=int,
    default=None,
    help='Number of classes in the custom model. If not provided and using checkpoint, will auto-detect from checkpoint. Default: 90 (COCO)'
)
args = parser.parse_args()

# Map model choice to model class
model_map = {
    'nano': RFDETRNano,
    'small': RFDETRSmall,
    'base': RFDETRBase,
    'medium': RFDETRMedium,
    'large': RFDETRLarge
}

# Initialize the selected model
print(f"Loading RF-DETR {args.model.upper()} model...")

# Auto-detect num_classes and class_names from checkpoint if available
num_classes = args.num_classes
class_names = None

if args.checkpoint:
    print(f"Loading custom weights from: {args.checkpoint}")
    try:
        # Try to load checkpoint metadata to detect num_classes and class_names
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'args' in checkpoint:
            checkpoint_args = checkpoint['args']
            if hasattr(checkpoint_args, 'num_classes'):
                detected_num_classes = checkpoint_args.num_classes
                if num_classes is None:
                    num_classes = detected_num_classes
                    print(f"Auto-detected {num_classes} classes from checkpoint")
                elif num_classes != detected_num_classes:
                    print(f"Warning: Specified --num-classes ({num_classes}) differs from checkpoint ({detected_num_classes}). Using specified value.")
            
            if hasattr(checkpoint_args, 'class_names'):
                class_names = checkpoint_args.class_names
                print(f"Using custom class names from checkpoint: {class_names}")
        elif num_classes is None:
            print("Warning: Could not detect num_classes from checkpoint. Using default (90 COCO classes).")
            print("If your model has a different number of classes, specify --num-classes")
    except Exception as e:
        print(f"Warning: Could not read checkpoint metadata: {e}")
        if num_classes is None:
            print("Using default (90 COCO classes). Specify --num-classes if needed.")
    
    # Initialize model with num_classes if specified
    if num_classes is not None:
        model = model_map[args.model](pretrain_weights=args.checkpoint, num_classes=num_classes)
    else:
        model = model_map[args.model](pretrain_weights=args.checkpoint)
else:
    print("Using pretrained weights...")
    model = model_map[args.model]()
    # Use COCO classes for pretrained models
    class_names = None

print(f"Model loaded successfully!")

# Use custom class names if available, otherwise use COCO_CLASSES
if class_names is not None:
    CLASSES = class_names
    print(f"Using custom classes: {CLASSES}")
else:
    CLASSES = COCO_CLASSES

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break

    detections = model.predict(frame[:, :, ::-1].copy(), threshold=0.2)

    labels = [
        f"{CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    cv2.imshow("Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()