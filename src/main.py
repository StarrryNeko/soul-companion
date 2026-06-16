"""Application entrypoint."""

from config.settings import UI_CONFIG
from src.pipeline import MentalHealthPipeline
from src.ui.gradio_app import create_app


def main() -> None:
    pipeline = MentalHealthPipeline()
    app = create_app(pipeline)
    app.launch(
        server_name=UI_CONFIG["server_name"],
        server_port=UI_CONFIG["server_port"],
        share=False,
    )


if __name__ == "__main__":
    main()

