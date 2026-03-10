from pathlib import Path
from core.detectors.image_detector import ImageDetector

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
        "AI SAMPLES": Path("test_samples/ai_samples"),
        "REAL SAMPLES": Path("test_samples/real_samples")
    }

    for category, folder in test_folders.items():
        print(f"\n{'=' * 80}")
        print(f"{category}")
        print(f"{'=' * 80}")

        for img_path in sorted(folder.glob("*")):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"\n📷 {img_path.name}")
                print("-" * 60)

                for name, detector in detectors.items():
                    result = detector.predict(img_path)
                    verdict = "AI" if result['class'] == 'AI-generated' else "REAL"
                    print( f"{name:10} | {verdict} | AI: {result['confidence_ai'] * 100:5.1f}% | Real: {result['confidence_real'] * 100:5.1f}%")
                print("-" * 60)