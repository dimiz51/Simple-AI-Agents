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
@click.option(
    "--respond-json",
    is_flag=True,
    default=False,
    help="Enable or disable JSON response format.",
)
def main(show_reasoning, model_name, respond_json):
    """
    CLI chat tool to interact with TubeMasterAgent asynchronously.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If a loop exists, create a background task
        print("Detected running event loop, using `asyncio.create_task()`")
        task = loop.create_task(chat_loop(show_reasoning, model_name, respond_json))
        loop.run_until_complete(task)  # Run until this task finishes
    else:
        # If no loop exists, start a fresh one
        asyncio.run(chat_loop(show_reasoning, model_name, respond_json))


async def chat_loop(show_reasoning, model_name, json_response=False):
    """Asynchronous chat loop to interact with TubeMasterAgent."""
    agent = TubeMasterAgent(model_id=model_name, respond_json=json_response)

    print("\nWelcome to Youtube Master Agent Chat! Type 'exit' to quit.\n")
    print(f"Using LLM model: {model_name}")
    print(f"Reasoning visible: {show_reasoning}\n")

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() in ["exit", "quit"]:
            print("******** Goodbye! See you next time! ********")
            break

        response = await agent.call_agent(prompt, show_reasoning=show_reasoning)

        print("\n******** Agent Response ********")
        print(response)
        print("\n")


if __name__ == "__main__":
    main()
