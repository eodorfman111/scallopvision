"""
Plain-English behavioral summary from the pipeline's stats dict.

Adapted from fishvision-demo/demo.py's generate_ai_summary /
generate_fallback_summary (dual Gemini/OpenAI-compatible client, graceful
no-key fallback to a rule-based summary).

Schema note: the source footage turned out to be multi-day timelapses, not
continuous sessions (see pipeline.py's module docstring), so there is no
per-scallop tracking or session dwell-time here - stats are aggregated
per-frame observations bucketed by that frame's own light color, across the
whole timelapse.

Metric note: the scallop population in each tank is fixed and enclosed - it
doesn't grow or shrink between frames. Raw observation totals/fractions are
NOT a fair comparison across colors, since a color sampled in more frames
would accumulate a bigger total at an identical per-frame rate. The valid
comparison is stats["light_color_avg_per_frame"] - average scallops detected
per frame OF THAT COLOR - which is what this module uses throughout. Do not
reintroduce light_color_observation_fractions as the headline number.
"""

import os

from openai import OpenAI

COLOR_LABELS = {
    "green": "green light",
    "blue": "blue light",
    "red": "red light (night)",
    "unknown": "indeterminate/unclear lighting",
}


def _sorted_colors(avgs: dict):
    return sorted(avgs.items(), key=lambda kv: kv[1], reverse=True)


def _preference_sentence(avgs: dict):
    real = {c: v for c, v in avgs.items() if c != "unknown"}
    if not real:
        return "not enough clearly-lit frames were available to draw a color comparison"
    ranked = _sorted_colors(real)
    if len(ranked) == 1:
        color, val = ranked[0]
        return f"the only comparable color was **{COLOR_LABELS.get(color, color)}**, averaging {val:.1f} scallops detected per frame"
    top_color, top_val = ranked[0]
    second_color, second_val = ranked[1]
    rel_diff = (top_val - second_val) / second_val if second_val else 0
    if rel_diff < 0.08:
        return (
            f"scallops showed no meaningful difference between {COLOR_LABELS.get(top_color, top_color)} "
            f"({top_val:.1f}/frame) and {COLOR_LABELS.get(second_color, second_color)} ({second_val:.1f}/frame)"
        )
    return (
        f"scallops were somewhat more often visible/favorably positioned under **{COLOR_LABELS.get(top_color, top_color)}**, "
        f"averaging {top_val:.1f} detected per frame versus {second_val:.1f} per frame under "
        f"{COLOR_LABELS.get(second_color, second_color)} — a {rel_diff * 100:.0f}% relative difference"
    )


def generate_fallback_summary(stats: dict, session_name: str = "this tank") -> str:
    cam_a = stats.get("camera_a", {})
    cam_b = stats.get("camera_b", {})
    avgs = stats.get("light_color_avg_per_frame", {})
    total_obs = stats.get("total_floor_observations", 0)
    total_frames = cam_a.get("frames_sampled", 0) + cam_b.get("frames_sampled", 0)

    breakdown_lines = "\n".join(
        f"- **{COLOR_LABELS.get(color, color)}**: {val:.1f} scallops detected per frame, on average"
        for color, val in _sorted_colors(avgs)
    )

    return f"""**Scallop Light-Preference Summary — {session_name}**

This analysis sampled {total_frames} independent frames across both camera angles ({cam_a.get('name', 'camera A')} and {cam_b.get('name', 'camera B')}), spanning a multi-day timelapse rather than one continuous sitting. The tank's scallops are a fixed, enclosed population - they don't enter or leave - so the {total_obs:.0f} total detections aren't a headcount of distinct animals, they're {total_frames} snapshots each catching some fraction of the same population. To compare colors fairly, each is measured as average scallops detected per frame *of that color*:

{breakdown_lines}

Overall, {_preference_sentence(avgs)}.

*(Rule-based summary — set an OPENAI_API_KEY or GEMINI_API_KEY environment variable for a fuller narrative write-up.)*"""


def generate_ai_summary(stats: dict, session_name: str = "this tank") -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return generate_fallback_summary(stats, session_name)

    try:
        if os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            model_name = "gemini-1.5-flash"
        else:
            client = OpenAI(api_key=api_key)
            model_name = "gpt-4o-mini"

        cam_a = stats.get("camera_a", {})
        cam_b = stats.get("camera_b", {})
        avgs = stats.get("light_color_avg_per_frame", {})

        prompt = f"""
        You are a marine biology research assistant summarizing a scallop light-preference tank study.

        Tank: {stats.get('tank')}
        Session label: {session_name}

        IMPORTANT context: the source footage is a multi-day timelapse (not one continuous
        recording), so this data has NO individual-scallop tracking and NO session dwell-time -
        every sampled frame is an independent snapshot. Each frame's own light color was
        classified on the spot (light color/position rotates over time in this tank setup).

        CRITICAL: the tank's scallop population is fixed and enclosed - scallops don't enter or
        leave between frames. So this data does NOT show "more scallops" appearing under one
        color - it shows how many of the same fixed population were visible/detectable/favorably
        positioned in a given frame. Never phrase this as scallops "appearing," "showing up," or
        "there being more of them" under a color - phrase it as detection/visibility/positioning.

        CRITICAL: use ONLY the average-detections-per-frame numbers below as the comparison metric.
        Do NOT use or compute percentages/fractions of a total - a color sampled in more frames
        would rack up a bigger raw total at an identical per-frame rate, which would misrepresent
        the comparison. The average-per-frame numbers already correct for that.

        Camera "{cam_a.get('name')}": {cam_a.get('frames_sampled', 0)} frames sampled
        Camera "{cam_b.get('name')}": {cam_b.get('frames_sampled', 0)} frames sampled

        Average scallops detected per frame, by light color, COMBINED across both cameras
        (the ONLY metric to compare colors with - no per-camera breakdown by color is given,
        so do not invent one): {avgs}

        Write a concise (under 180 words) plain-English summary of what this data suggests, e.g.
        "Frames lit green averaged X scallops detected, versus Y under blue and Z under red." If the
        differences are small (under ~15% relative), say explicitly that there's no meaningful
        difference rather than dressing up noise as a preference. Frame this as an aggregate
        spatial/visibility tendency across many independent snapshots of one fixed population, not
        proof that any individual scallop "chose" or "moved toward" a color - the data can't support
        animal-level or movement claims, only population-level detection frequency.

        IMPORTANT: refer to the two cameras by their actual names given above (e.g. "{cam_a.get('name')}"
        and "{cam_b.get('name')}"), never as generic "Camera A" / "Camera B".
        """

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a marine biology research assistant writing careful, statistically honest behavioral summaries for a scallop light-preference study, from timelapse snapshot data (no individual tracking, fixed enclosed population)."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        fallback = generate_fallback_summary(stats, session_name)
        return f"{fallback}\n\n*(Note: LLM summary generation failed ({e}). Displaying rule-based fallback.)*"
