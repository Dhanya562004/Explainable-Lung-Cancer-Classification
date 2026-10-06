import os
from src.config import PROMOTION_CRITERIA, DEFAULT_MODEL_PATH
from src.models.model_registry import register_model
from src.evaluation.evaluate import evaluate_model

def evaluate_and_promote(candidate_model_path=DEFAULT_MODEL_PATH, new_version_name="v1.1.0-xception-retrained"):
    """
    Evaluates candidate retrained model against production baseline criteria.
    Promotes and registers candidate only if validation thresholds are satisfied.
    """
    print(f"[INFO] Evaluating candidate model for promotion: {candidate_model_path}")
    
    if not os.path.exists(candidate_model_path):
        return {
            'promoted': False,
            'reason': f"Candidate model file does not exist at {candidate_model_path}"
        }

    metrics = evaluate_model(model_path=candidate_model_path)
    
    macro_f1 = metrics.get('macro_f1_score', 0.0)
    accuracy = metrics.get('test_accuracy', 0.0)
    
    min_f1 = PROMOTION_CRITERIA['minimum_macro_f1']
    min_acc = PROMOTION_CRITERIA['minimum_accuracy']

    if macro_f1 >= min_f1 and accuracy >= min_acc:
        reg_entry = register_model(
            version=new_version_name,
            architecture="Xception Transfer Learning (Retrained)",
            metrics=metrics,
            training_config={"promotion": "passed_criteria", "min_macro_f1": min_f1},
            model_path=candidate_model_path,
            set_active=True
        )
        print(f"[PROMOTION APPROVED] Candidate model promoted to active version '{new_version_name}'.")
        return {
            'promoted': True,
            'version': new_version_name,
            'metrics': metrics,
            'registry_entry': reg_entry
        }
    else:
        reason = (
            f"Candidate metrics (F1: {macro_f1:.4f}, Acc: {accuracy:.4f}) failed to meet "
            f"minimum criteria (F1 >= {min_f1}, Acc >= {min_acc}). Promotion rejected."
        )
        print(f"[PROMOTION REJECTED] {reason}")
        return {
            'promoted': False,
            'reason': reason,
            'metrics': metrics
        }

if __name__ == '__main__':
    evaluate_and_promote()
