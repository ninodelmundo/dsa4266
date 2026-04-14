#!/usr/bin/env python3
"""Serve a saved Shapash SmartExplainer as an interactive local dashboard."""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Serve a saved Shapash dashboard.")
    parser.add_argument(
        "--explainer",
        default="outputs/explainability/shapash_explainer.pkl",
        help="Path to the saved Shapash SmartExplainer pickle.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local dashboard.")
    parser.add_argument("--port", type=int, default=8050, help="Port for the local dashboard.")
    args = parser.parse_args()

    try:
        from shapash import SmartExplainer
    except ImportError as exc:
        raise SystemExit(
            "Shapash is not importable in this Python environment.\n"
            "Install it in the active environment with:\n"
            "  python3 -m pip install shapash dash\n"
            "Then verify with:\n"
            "  python3 -c \"import shapash; print(shapash.__version__)\""
        ) from exc

    xpl = SmartExplainer.load(args.explainer)
    url = f"http://{args.host}:{args.port}"
    print(f"Starting Shapash dashboard at {url}")
    print("Press Ctrl+C to stop the dashboard server.")
    app_thread = xpl.run_app(host=args.host, port=args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if hasattr(app_thread, "kill"):
            app_thread.kill()
        print("\nStopped Shapash dashboard.")


if __name__ == "__main__":
    main()
