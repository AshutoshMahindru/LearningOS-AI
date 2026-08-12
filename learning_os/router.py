"""Simple routing logic placeholder for adaptive learning."""


def next_action(mission, blocked=False):
    if blocked:
        return {
            "action": "zoom_in",
            "reason": "resolve prerequisite blocker",
            "return_to": mission,
        }

    return {
        "action": "continue_mission",
        "mission": mission,
    }
