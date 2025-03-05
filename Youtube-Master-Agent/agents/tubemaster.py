import asyncio
import dotenv
import os
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream
from llama_index.core.workflow import Context

# Import tools
from tools.response_formatter import video_information_response
from tools.transcriber import transcribe_audio
from tools.youtube_fetcher import download_youtube_audio

# Load environment variables
dotenv.load_dotenv()


class TubeMasterAgent:
    """
    A class-based YouTube video summarization agent that downloads,
    transcribes, and summarizes YouTube videos.
    """

    def __init__(self, model_id="Qwen/Qwen2.5-Coder-32B-Instruct"):
        """
        Initializes the agent with the required tools and system prompt.
        """

        # Load the HF token
        token = os.getenv("HF_API_KEY")
        if token is None:
            raise ValueError("Please set the HF_API_KEY environment variable.")

        # Large Language Model(LLM)
        self.llm = HuggingFaceInferenceAPI(model_name=model_id, token=token)

        self.agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[
                transcribe_audio,
                download_youtube_audio,
                video_information_response,
            ],
            llm=self.llm,
            system_prompt=(
                "You are an AI assistant that can use tools to transcribe audio from YouTube videos, "
                "create summaries of the video's content, provide contextual information, and answer related questions. "
                "You will NEVER respond to tasks unrelated to your role; instead, alert the user that you cannot perform the task. "
                "ALWAYS provide summaries, titles, video topics, and video URLs for all successfully transcribed videos. "
                "For failed transcriptions, return only successful results along with a detailed explanation of errors. "
                "Use available tools to format the response effectively."
            ),
        )

        # Agent's memory
        self.context = Context(self.agent)

    async def call_agent(self, video_prompt: str, show_reasoning: bool = False) -> str:
        """
        Asynchronously processes a string prompt containing YouTube video URLs.
        """

        # Run the agent (Wraps the LlamaIndex AgentWorkflow run method)
        handler = self.agent.run(video_prompt, ctx=self.context)

        # Stream events and capture responses if needed
        if show_reasoning:
            print()
            print("********Agent Reasoning********")
            async for ev in handler.stream_events():
                if isinstance(ev, ToolCallResult):
                    print(
                        f"\nCalled tool: {ev.tool_name} {ev.tool_kwargs} => {ev.tool_output}"
                    )
                elif isinstance(ev, AgentStream):
                    print(ev.delta, end="", flush=True)
            print()

        # Get the final response
        response = await handler

        response_text = response.response.blocks[-1].text

        return response_text

    def run(self, video_prompt: str, show_reasoning: bool = False) -> str:
        """
        Runs the async function inside an event loop for synchronous execution.
        """
        return asyncio.run(self.call_agent(video_prompt, show_reasoning=show_reasoning))
