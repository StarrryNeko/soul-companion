"""Response generator tests."""

from src.model.generator import ResponseGenerator


class FailingClient:
    model = "deepseek-chat"

    def generate(self, *args, **kwargs) -> str:
        raise AssertionError("external API should not be called in template mode")


def test_generator_can_force_template_backend() -> None:
    generator = ResponseGenerator(fallback_client=FailingClient())
    generator.set_chat_model("template")

    response = generator.generate("hello")

    assert generator.last_backend == "template"
    assert response


def test_generator_can_switch_external_model_name() -> None:
    client = FailingClient()
    generator = ResponseGenerator(fallback_client=client)

    generator.set_chat_model("deepseek_api", "deepseek-reasoner")

    assert generator.backend_preference == "deepseek_api"
    assert client.model == "deepseek-reasoner"
