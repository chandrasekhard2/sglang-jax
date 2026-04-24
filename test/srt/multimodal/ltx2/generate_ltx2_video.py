import os
import requests
import time
import subprocess
import sys

from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    LTX2_MODEL,
    popen_launch_server,
)

def run_test():
    process = popen_launch_server(
        LTX2_MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        device="tpu",
        other_args=[
            "--trust-remote-code",
            "--skip-server-warmup",
            "--random-seed", "42",
            "--multimodal",
            "--tokenizer-path", "Lightricks/LTX-2",
            "--download-dir", "/mnt/disks/persist/hf_cache",
            "--tp-size", "4",
            "--sp-size", "2",
            "--enable-single-process",
            "--max-total-tokens", "32768",
            "--context-length", "8192",
        ],
        env={
            "HF_HOME": "/mnt/disks/persist/hf_cache",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
        multimodal=True,
    )
    
    try:
        print("Server started. Sending request...")
        data = {
            "prompt": "Style: animated cinematic shot, realistic with cinematic lighting. In a medium shot, a heavy metallic robot walks slowly forward. The camera dollies back, keeping the robot's slow, deliberate walk perfectly in frame. The audio features rhythmic, heavy metallic clanking and whirring of gears as the robot steps on the ground. Suddenly, the robot starts running slowly and heavily, its footfalls turning into loud, resonant thuds against the ground, accompanied by a rising hum of its engine. It then stops abruptly, its metal joints squeaking to a halt with a heavy thud. The camera keeps dollying back, revealing a blue, similar robot appearing in an over-the-shoulder shot. The scene is filled with ambient industrial sounds, soft wind, and the echoing, metallic footsteps of the robots, with no music playing.",
            "neg_prompt": "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.",
            "num_frames": 121,
            "size": "1280*704", # 720P resolution
            "num_inference_steps": 40,
            "guidance_scale": 3.0,
            "stg_scale": 1.0,
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/api/v1/videos/generation",
            headers=headers,
            json=data,
            timeout=1200, 
        )
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            print("SUCCESS: Video generated.")
        else:
            print("FAILED:", result)
    except Exception as e:
        print("Error during request:", e)
        if hasattr(e, "response") and e.response:
            print("Response:", e.response.text)
    finally:
        process.kill()

if __name__ == "__main__":
    run_test()
