"""Visual diagnostics via Gemini VLM.

Captures a screenshot of the Isaac Lab simulation and asks Gemini to analyze
physical plausibility — things a human would catch instantly but programmatic
checks miss (initial pose sanity, grasp geometry, manipulation feasibility).

Setup: place API key at ``.key/gemini_key.json`` as ``{"api_key": "..."}``.

Usage:
    from rl_isaaclab.diagnostics.vlm import analyze_image, visual_check
    report = analyze_image("screenshot.png", "Is the hand grasping the object?")
    print(report)
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# --- auth ---
_REPO_ROOT = Path(__file__).resolve().parents[2]
_API_KEY_PATH = _REPO_ROOT / ".key" / "gemini_key.json"


def _get_client():
    """Create Gemini client from API key JSON."""
    from google import genai

    if not _API_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Gemini API key not found at {_API_KEY_PATH}. "
            "Create it with: {\"api_key\": \"YOUR_KEY\"}"
        )
    with open(_API_KEY_PATH) as f:
        api_key = json.load(f).get("api_key", "")
    if not api_key:
        raise ValueError(f"Empty api_key field in {_API_KEY_PATH}")
    return genai.Client(api_key=api_key)


# --- models ---
# Picked from .key/gemini_available_models.json (2026-03-17 snapshot):
# - PRO:       deep reasoning, reward/config diagnosis (~60s)
# - FLASH:     standard visual diagnostics (~1.5s)
# - FLASH_LITE: fastest binary pass/fail (~1s)
# - ROBOTICS:  embodied manipulation specialist
MODEL_PRO = "gemini-3.1-pro-preview"
MODEL_FLASH = "gemini-2.5-flash"
MODEL_FLASH_LITE = "gemini-2.5-flash-lite"
MODEL_ROBOTICS = "gemini-robotics-er-1.5-preview"


def analyze_image(
    image_path: str | Path,
    prompt: str,
    model: str = MODEL_FLASH,
) -> str:
    """Send an image to Gemini for analysis.

    Args:
        image_path: Path to image file (PNG or JPEG).
        prompt: Text prompt describing what to analyze.
        model: Gemini model name. Defaults to FLASH (fast, cheap).

    Returns:
        Gemini's text response.
    """
    from google.genai import types

    client = _get_client()
    path = Path(image_path)
    image_bytes = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
    )
    return response.text


def analyze_images(
    image_paths: list[str | Path],
    prompt: str,
    model: str = MODEL_FLASH,
) -> str:
    """Send multiple images (e.g., a rollout sequence) to Gemini.

    Args:
        image_paths: List of PNG/JPEG paths (order preserved).
        prompt: Text prompt.
        model: Gemini model.

    Returns:
        Gemini's text response.
    """
    from google.genai import types

    client = _get_client()
    contents = []
    for p in image_paths:
        path = Path(p)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
    contents.append(prompt)

    response = client.models.generate_content(model=model, contents=contents)
    return response.text


# --- prompts ---

_INIT_POSE_PROMPT = """\
You are a robotics researcher specializing in dexterous manipulation. You are
analyzing an Isaac Lab simulation screenshot showing a {robot_name} robot hand
with a cylinder for in-hand rotation RL training.

IMPORTANT CONTEXT:
- This is the INITIAL pose before the policy acts — it's a pre-computed stable grasp.
- The initial grasp determines whether RL training can succeed at all.
- For in-hand rotation, we need ≥3 fingers making contact to form a manipulation cage.
- The hand rotation/position relative to the object must be physically sensible.

Provide a structured report:

## 1. Scene Check
- Is the hand visible and in a natural orientation?
- Is the cylinder object visible?
- Is the cylinder near the palm/fingers, or floating far away?
- Any obvious interpenetration (finger meshes inside cylinder mesh)?

## 2. Contact Analysis
For each visible finger (thumb, index, middle, ring, pinky), classify:
- CONTACT: fingertip visibly touching cylinder
- NEAR (<1cm gap): close but not touching
- FAR: too far to contact
Count fingers in contact. Is this a viable grasp for in-hand rotation (need ≥3)?

## 3. Grasp Geometry
- What grasp type? (fingertip cage / power grasp / palmar / precision pinch)
- Is the cylinder supported against gravity?
- Is thumb opposing the fingers (good for rotation) or on same side?

## 4. Red Flags
Any of these with severity [CRITICAL / WARNING / MINOR]:
- Cylinder floating with no support
- Finger-cylinder interpenetration
- Cylinder too large/small for finger span
- Hand in impossible pose (joints at limits)
- Cylinder outside reachable workspace
- Wrong root hand orientation (palm-down when should be palm-up, etc.)

## 5. Verdict
One line: TRAINABLE / MARGINAL / NOT TRAINABLE
Brief justification.

Be specific. If you can estimate distances, do so in millimeters.
"""


def check_initial_pose(
    image_path: str | Path,
    robot_name: str = "SharpaWave",
    model: str = MODEL_PRO,
) -> str:
    """Check whether an initial hand+object pose is a viable starting state
    for in-hand rotation RL training.

    Args:
        image_path: PNG/JPEG of the scene (at t=0 or after a few settling steps).
        robot_name: Name of the robot hand (SharpaWave or xhand).
        model: Gemini model. Defaults to ROBOTICS (manipulation specialist).

    Returns:
        Structured analysis report.
    """
    return analyze_image(image_path, _INIT_POSE_PROMPT.format(robot_name=robot_name), model=model)


_ROLLOUT_PROMPT = """\
You are analyzing a {n}-frame sequence from a dexterous manipulation RL
rollout in Isaac Lab. The robot ({robot_name}) is tasked with rotating a
cylinder around its Z-axis (yaw) in-hand.

{extra_context}

## 1. Motion Check
- Is the hand/fingers moving between frames, or static?
- Is the cylinder rotating, translating, or slipping?
- Estimate the object rotation across all frames (degrees around Z).

## 2. Manipulation Strategy
What does the policy seem to be doing?
- Finger gaiting (sequential lift-reposition-press)
- Rolling (simultaneous coordinated finger motion)
- Pivoting around a single contact
- Pushing against palm
- No coherent strategy (flailing)

## 3. Failure Mode (if any)
- Object slipping / dropping
- Fingers oscillating without progress
- Some fingers ignored by policy
- Object in unrecoverable state

## 4. Verdict
One line: GOOD / PARTIAL / BAD rotation behavior.
Brief justification.
"""


def analyze_rollout(
    image_paths: list[str | Path],
    robot_name: str = "SharpaWave",
    extra_context: str = "",
    model: str = MODEL_PRO,
) -> str:
    """Analyze a multi-frame rollout sequence to diagnose policy behavior.

    Args:
        image_paths: Ordered list of PNG frames.
        robot_name: Robot hand name.
        extra_context: Optional training-step / reward info ("step 100M, reward 800").
        model: Gemini model. PRO default for deeper reasoning.

    Returns:
        Rollout analysis.
    """
    prompt = _ROLLOUT_PROMPT.format(
        n=len(image_paths),
        robot_name=robot_name,
        extra_context=(f"Context: {extra_context}" if extra_context else ""),
    )
    return analyze_images(image_paths, prompt, model=model)


# Legacy-compatible wrappers kept for reference ------------------------------

_LEGACY_ENV_CHECK_PROMPT = """\
You are a robotics researcher specializing in dexterous manipulation. You are
analyzing an Isaac Lab simulation screenshot showing a robot hand manipulating
an object.

IMPORTANT CONTEXT:
- This is RL training setup — the hand starts in a fixed pose before the policy acts.
- The initial grasp configuration determines whether training can succeed at all.
- A grasp that looks "close" but has no actual fingertip-object contact is USELESS for RL.
- For in-hand reorientation, you need ≥3 fingers making simultaneous contact.

Analyze and provide a structured report with Contact Analysis, Grasp Geometry,
Manipulation Feasibility, Red Flags, and one-line Verdict (TRAINABLE / MARGINAL
/ NOT TRAINABLE).
"""


def visual_check(
    image_path: str | Path,
    task_name: str = "inhand_rotate",
    robot_name: str = "SharpaWave",
    model: str = MODEL_PRO,
) -> str:
    """High-level visual plausibility check on a simulation screenshot."""
    prompt = f"Task: {task_name}\nRobot: {robot_name}\n\n" + _LEGACY_ENV_CHECK_PROMPT
    return analyze_image(image_path, prompt, model=model)


def analyze_training_frame(
    image_path: str | Path,
    context: str = "",
    model: str = MODEL_PRO,
) -> str:
    """Analyze a single frame from a training rollout for diagnosis."""
    prompt = f"""\
You are a robotics RL researcher analyzing a frame from a dexterous manipulation
training rollout in Isaac Lab simulation.

{f'Context: {context}' if context else ''}

Diagnose finger-object state, policy strategy, failure mode (if any), and give
actionable recommendations for reward/init/curriculum/action-space changes.
Be quantitative where possible (mm, degrees).
"""
    return analyze_image(image_path, prompt, model=model)


def suggest_improvements(
    image_path: str | Path,
    task_name: str = "inhand_rotate",
    current_config: str = "",
    training_stats: str = "",
    model: str = MODEL_PRO,
) -> str:
    """Ask Gemini for prioritized improvement suggestions."""
    prompt = f"""\
You are an expert RL researcher for dexterous manipulation. Given the current
simulation state and training progress, propose specific improvements.

Task: {task_name}
{f'Current config:{chr(10)}{current_config}' if current_config else ''}
{f'Training stats:{chr(10)}{training_stats}' if training_stats else ''}

For each suggestion provide: what to change, why, expected impact, risk.
Prioritize highest-impact / lowest-risk. Be specific with numbers.
Limit to top 3-5 suggestions.
"""
    return analyze_image(image_path, prompt, model=model)
