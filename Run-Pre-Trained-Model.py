import argparse
import cv2
import supervision as sv
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
if args.checkpoint:
    print(f"Loading custom weights from: {args.checkpoint}")
    model = model_map[args.model](pretrain_weights=args.checkpoint)
else:
    print("Using pretrained weights...")
    model = model_map[args.model]()
print(f"Model loaded successfully!")

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break

    detections = model.predict(frame[:, :, ::-1].copy(), threshold=0.5)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
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