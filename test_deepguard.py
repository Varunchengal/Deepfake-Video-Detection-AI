
import torch
import cv2
from inference.video_predictor import VideoPredictor

VIDEO_PATH = "test_video.mp4"

def main():
    device = torch.device("cpu")
    torch.set_num_threads(2)

    print("Loading DeepGuard model...")

    predictor = VideoPredictor(
        margin_ratio=0.2,
        conf_thres=0.5,
        min_face_ratio=0.01,
        model_name="ms_eff_gcvit_b0",
        dataset="celeb_df_v2"
    )

    print("Model loaded.")
    print("Analysing video...")

    if not cv2.VideoCapture(VIDEO_PATH).isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    fake_probability = predictor.predict_video(
        video_path=VIDEO_PATH,
        num_frames=8,
        agg_mode="conf",
        tta_hflip=0.0
    )

    fake_probability = float(fake_probability)

    print("\n========== RESULT ==========")
    print(f"Fake probability: {fake_probability:.2%}")
    print(f"Real probability: {1 - fake_probability:.2%}")

    if fake_probability >= 0.5:
        print("Prediction: FAKE")
    else:
        print("Prediction: REAL")

    print("============================")


if __name__ == "__main__":
    main()
