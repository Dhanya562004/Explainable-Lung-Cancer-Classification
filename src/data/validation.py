from src.data.ingestion import scan_dataset

def validate_dataset(dataset_dir=None):
    """
    Validates dataset directory integrity and returns diagnostic report.
    Returns status: HEALTHY, WARNING, or CRITICAL.
    """
    summary = scan_dataset(dataset_dir) if dataset_dir else scan_dataset()

    errors = []
    warnings = []

    if not summary['exists']:
        errors.append(f"Dataset root directory missing: {summary['dataset_dir']}")

    for split in ['train', 'valid', 'test']:
        if split not in summary['splits'] or not summary['splits'][split]['exists']:
            errors.append(f"Required split directory missing: '{split}'")
        else:
            if summary['splits'][split]['total'] == 0:
                warnings.append(f"Split directory '{split}' contains 0 images.")

    if summary['missing_classes']:
        errors.append(f"Missing class subdirectories: {', '.join(summary['missing_classes'])}")

    if summary['corrupt_images'] > 0:
        warnings.append(f"Detected {summary['corrupt_images']} unreadable/corrupt image file(s).")

    if summary['total_images'] < 50:
        warnings.append(f"Dataset sample count ({summary['total_images']}) is very small.")

    status = "HEALTHY"
    if errors:
        status = "CRITICAL"
    elif warnings:
        status = "WARNING"

    return {
        'status': status,
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'summary': summary
    }

if __name__ == '__main__':
    val_report = validate_dataset()
    print("Validation Status:", val_report['status'])
    print("Is Valid:", val_report['is_valid'])
    print("Errors:", val_report['errors'])
    print("Warnings:", val_report['warnings'])
