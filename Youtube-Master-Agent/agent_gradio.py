import gradio as gr
import asyncio
from agents.tubemaster import TubeMasterAgent

# 🔹 Global agent instance (default settings)
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
global_agent = TubeMasterAgent(model_id=DEFAULT_MODEL, respond_json=False)

# 🔹 Create a global event loop and start it immediately
global_loop = asyncio.get_event_loop()

# Start the event loop if it's not running already (only needs to run once)
if not global_loop.is_running():
    asyncio.set_event_loop(global_loop)


async def generate_response(message, history, show_reasoning):
    """Handles responses from the global TubeMasterAgent asynchronously."""
    response = await global_agent.call_agent(message, show_reasoning=show_reasoning)
    return response


def respond(message, history, show_reasoning):
    """Handles multiple requests using a shared event loop."""
    # If the global loop is running, use it to create a task
    task = global_loop.create_task(generate_response(message, history, show_reasoning))
    return global_loop.run_until_complete(task)


# 🔹 UI Components
demo = gr.ChatInterface(
    respond,
    additional_inputs=[gr.Checkbox(value=False, label="Show Reasoning in CLI")],
    title="TubeMaster AI Agent",
    description="An agent that can transcribe, summarize and understand YouTube video content for Q&As, and more!",
)

if __name__ == "__main__":
    demo.launch()
