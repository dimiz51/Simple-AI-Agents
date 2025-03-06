import click
import asyncio
from agents.tubemaster import TubeMasterAgent


@click.command()
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
def main(show_reasoning, model_name):
    """
    CLI chat tool to interact with TubeMasterAgent asynchronously.
    """
    asyncio.run(chat_loop(show_reasoning, model_name))


async def chat_loop(show_reasoning, model_name):
    """Asynchronous chat loop to interact with TubeMasterAgent."""
    agent = TubeMasterAgent(model_id=model_name)

    print("\nWelcome to Youtube Master Agent Chat! Type 'exit' to quit.\n")
    print(f"Using LLM model: {model_name}")
    print(f"Reasoning visible: {show_reasoning}\n")

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() in ["exit", "quit"]:
            print("Hope I helped you find what you were looking for! Goodbye!")
            break

        response = await agent.call_agent(prompt, show_reasoning=show_reasoning)

        print("\n******** Agent Response ********")
        print(response)
        print("\n")


if __name__ == "__main__":
    main()
