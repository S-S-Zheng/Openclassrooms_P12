
# Imports
import os
import logging

import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def smoke_test():
    logger.info("Smode test en cours...")
    model_path = "datas/results/best_model/best_model.onnx"
    if not os.path.exists(model_path):
        logger.warning(f"Echeck du smoke test -- Modèle introuvable à {model_path}")
        exit(1)
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        logger.info(f"Smoke test réussi -- Modèle chargé. Input attendu : {input_name}")
    except Exception as e:
        logger.error(f"Erreur critique au chargement ONNX : {e}")
        exit(1)


if __name__ == "__main__":
    smoke_test()