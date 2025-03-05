import click
from agents.tubemaster import TubeMasterAgent


@click.command()
@click.argument("prompt", type=str)
@click.option(
    "--show-reasoning",
    is_flag=True,
    default=False,
    help="Enable or disable agent reasoning output.",
)
@click.option(
    "--model-name",
    default="Qwen/Qwen2.5-Coder-32B-Instruct",
    show_default=True,
    help="Specify the model name.",
)
def main(prompt, show_reasoning, model_name):
    """
    CLI tool to summarize YouTube videos using TubeMasterAgent.

    PROMPT: The text prompt specifying the videos to summarize.
    """
    agent = TubeMasterAgent(model_id=model_name)

    response = agent.run(prompt, show_reasoning=show_reasoning)

    print("\n******** Agent Response ********")
    print(response)


if __name__ == "__main__":
    main()
