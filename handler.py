import base64
import os
import shutil
import tempfile
import uuid
from argparse import Namespace
from pathlib import Path

import requests
import runpod
import yaml

from scripts.inference import main as run_inference


APP_DIR = Path("/app")
DEFAULT_RESULTS_DIR = APP_DIR / "results" / "serverless"


def _download_file(url: str, output_path: Path) -> Path:
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_obj.write(chunk)

    return output_path


def _build_args(config_path: Path, result_dir: Path, output_name: str, job_input: dict) -> Namespace:
    return Namespace(
        ffmpeg_path=job_input.get("ffmpeg_path", "/usr/bin"),
        gpu_id=int(job_input.get("gpu_id", 0)),
        vae_type=job_input.get("vae_type", "sd-vae"),
        unet_config=job_input.get("unet_config", "./models/musetalkV15/musetalk.json"),
        unet_model_path=job_input.get("unet_model_path", "./models/musetalkV15/unet.pth"),
        whisper_dir=job_input.get("whisper_dir", "./models/whisper"),
        inference_config=str(config_path),
        bbox_shift=int(job_input.get("bbox_shift", 0)),
        result_dir=str(result_dir),
        extra_margin=int(job_input.get("extra_margin", 10)),
        fps=int(job_input.get("fps", 25)),
        audio_padding_length_left=int(job_input.get("audio_padding_length_left", 2)),
        audio_padding_length_right=int(job_input.get("audio_padding_length_right", 2)),
        batch_size=int(job_input.get("batch_size", 8)),
        output_vid_name=output_name,
        use_saved_coord=bool(job_input.get("use_saved_coord", False)),
        saved_coord=bool(job_input.get("saved_coord", False)),
        use_float16=bool(job_input.get("use_float16", True)),
        parsing_mode=job_input.get("parsing_mode", "jaw"),
        left_cheek_width=int(job_input.get("left_cheek_width", 90)),
        right_cheek_width=int(job_input.get("right_cheek_width", 90)),
        version=job_input.get("version", "v15"),
    )


def _encode_base64(file_path: Path) -> str:
    with file_path.open("rb") as file_obj:
        return base64.b64encode(file_obj.read()).decode("utf-8")


def handler(job):
    job_input = job.get("input", {})
    video_url = job_input.get("video_url")
    audio_url = job_input.get("audio_url")

    if not video_url or not audio_url:
        return {
            "error": "Both 'video_url' and 'audio_url' are required."
        }

    job_id = job.get("id") or str(uuid.uuid4())
    work_dir = Path(tempfile.mkdtemp(prefix=f"musetalk_{job_id}_"))

    try:
        video_path = _download_file(video_url, work_dir / "input" / "video.mp4")
        audio_path = _download_file(audio_url, work_dir / "input" / "audio.wav")

        config_path = work_dir / "inference.yaml"
        output_name = job_input.get("output_name", f"{job_id}.mp4")
        result_dir = DEFAULT_RESULTS_DIR / job_id
        result_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "task_0": {
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "result_name": output_name,
            }
        }

        if "bbox_shift" in job_input:
            config_data["task_0"]["bbox_shift"] = int(job_input["bbox_shift"])

        with config_path.open("w", encoding="utf-8") as file_obj:
            yaml.safe_dump(config_data, file_obj, sort_keys=False)

        args = _build_args(config_path, result_dir, output_name, job_input)
        run_inference(args)

        output_path = result_dir / args.version / output_name
        if not output_path.exists():
            return {
                "error": f"Inference finished but output file was not found: {output_path}"
            }

        response = {
            "status": "success",
            "output_path": str(output_path),
            "job_id": job_id,
        }

        if bool(job_input.get("return_base64", False)):
            response["video_base64"] = _encode_base64(output_path)

        return response
    except Exception as exc:
        return {
            "error": str(exc),
            "job_id": job_id,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
