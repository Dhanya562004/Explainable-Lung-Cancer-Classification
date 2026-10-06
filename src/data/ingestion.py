import os
from PIL import Image
from src.config import DEFAULT_DATASET_DIR, CLASS_NAMES, CLASS_LABELS

def scan_dataset(dataset_dir=DEFAULT_DATASET_DIR):
    """
    Scans train/validation/test directories, identifies class labels,
    counts images, detects unreadable/corrupt files, measures image dimensions,
    and returns a structured dataset summary dictionary.
    """
    summary = {
        'dataset_dir': dataset_dir,
        'exists': os.path.exists(dataset_dir),
        'splits': {},
        'total_images': 0,
        'corrupt_images': 0,
        'missing_classes': [],
        'dimensions_sample': [],
        'class_distribution': {}
    }

    if not summary['exists']:
        return summary

    splits = ['train', 'valid', 'test']
    total_count = 0
    corrupt_count = 0
    dimensions_set = set()

    # Aggregate class breakdown across all splits
    class_totals = {cls_name: 0 for cls_name in CLASS_NAMES}

    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        summary['splits'][split] = {
            'exists': os.path.exists(split_dir),
            'total': 0,
            'classes': {}
        }

        if not summary['splits'][split]['exists']:
            continue

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                if cls_name not in summary['missing_classes']:
                    summary['missing_classes'].append(cls_name)
                summary['splits'][split]['classes'][cls_name] = 0
                continue

            valid_images_in_cls = 0
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if not os.path.isfile(fpath):
                    continue

                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    continue

                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    # Re-open after verify to get size safely
                    with Image.open(fpath) as img:
                        dimensions_set.add(img.size)  # (width, height)
                    valid_images_in_cls += 1
                except Exception:
                    corrupt_count += 1

            summary['splits'][split]['classes'][cls_name] = valid_images_in_cls
            summary['splits'][split]['total'] += valid_images_in_cls
            class_totals[cls_name] += valid_images_in_cls
            total_count += valid_images_in_cls

    summary['total_images'] = total_count
    summary['corrupt_images'] = corrupt_count
    summary['dimensions_sample'] = [f"{w}x{h}" for (w, h) in sorted(list(dimensions_set))[:5]]

    if total_count > 0:
        summary['class_distribution'] = {
            cls_name: {
                'count': count,
                'percentage': round((count / total_count) * 100, 2)
            }
            for cls_name, count in class_totals.items()
        }

    return summary

if __name__ == '__main__':
    report = scan_dataset()
    print("Dataset Summary:")
    print(f"Total Images: {report['total_images']}")
    print(f"Corrupt Images: {report['corrupt_images']}")
    print(f"Dimensions Sample: {report['dimensions_sample']}")
    print("Class Distribution:", report['class_distribution'])
