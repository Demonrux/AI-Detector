from pathlib import Path
from core.detectors.image_detector import ImageDetector
from  core.utils.logger import setup_logger

setup_logger()

if __name__ == "__main__":
    models_dir = Path(__file__).parent.parent / "core" / "models"

    detectors = {
        "ADM": ImageDetector(model_path=models_dir / "adm.pkl"),
        "Midjourney": ImageDetector(model_path=models_dir / "midjourney.pkl"),
        "Glide": ImageDetector(model_path=models_dir / "glide.pkl"),
        "BigGAN": ImageDetector(model_path=models_dir / "biggan.pkl"),
        "Wukong": ImageDetector(model_path=models_dir / "wukong.pkl"),
        "VQDM": ImageDetector(model_path=models_dir / "vqdm.pkl"),
    }

    test_folders = {
        "AI SAMPLES": Path("img/ai"),
        "REAL SAMPLES": Path("img/nature")
    }

    for category, folder in test_folders.items():
        print(f"\n{'=' * 80}")
        print(f"{category}")
        print(f"{'=' * 80}")

        for img_path in sorted(folder.glob("*")):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"\n{img_path.name}")
                print("-" * 70)

                ai_confs = []
                real_confs = []

                for name, detector in detectors.items():
                    result = detector.predict(img_path)
                    verdict = "AI" if result['class'] == 'AI-generated' else "REAL"
                    ai_conf = result['confidence_ai'] * 100
                    real_conf = result['confidence_real'] * 100

                    ai_confs.append(ai_conf)
                    real_confs.append(real_conf)

                    print(f"{name:10} | {verdict} | AI: {ai_conf:5.1f}% | Real: {real_conf:5.1f}%")

                avg_ai = sum(ai_confs) / len(ai_confs)
                avg_real = sum(real_confs) / len(real_confs)

                print("-" * 70)
                print(f"{'Average':10} | AI: {avg_ai:5.1f}% | Real: {avg_real:5.1f}%")
                print("-" * 70)