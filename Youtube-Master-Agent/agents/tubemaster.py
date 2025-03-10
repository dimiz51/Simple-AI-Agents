import asyncio
import dotenv
import os
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate


# Import tools
from tools.response_formatter import json_response_formatter
from tools.transcriber import transcribe_audio
from tools.youtube_fetcher import download_youtube_audio

# Load environment variables
dotenv.load_dotenv()


class TubeMasterAgent:
    """
    A class-based YouTube video summarization agent that downloads,
    transcribes, and summarizes YouTube videos.
    """

    def __init__(self, model_id="Qwen/Qwen2.5-Coder-32B-Instruct", respond_json=False):
        """
        Initializes the agent with the required tools and system prompt.
        """

        # Load the HF token
        token = os.getenv("HF_API_KEY")
        if token is None:
            raise ValueError("Please set the HF_API_KEY environment variable.")

        # Large Language Model(LLM)
        self.llm = HuggingFaceInferenceAPI(model_name=model_id, token=token)

        # Create the response formatting tool for video summaries
        response_format_tool = FunctionTool.from_defaults(
            async_fn=json_response_formatter,
            name="video_summary_response_formatter",
            description=(
                "Provides a JSON-like formatted response with the summaries, titles, topics, URLs for all videos the user asked for."
            ),
            return_direct=True if respond_json else False,
        )

        self.agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[
                transcribe_audio,
                download_youtube_audio,
                response_format_tool,
            ],
            llm=self.llm,
            system_prompt=(
                "You are an AI assistant that can use tools to transcribe audio from YouTube videos, "
                "create summaries of the video's content, provide contextual information, or can directly answer related questions. "
                "Before you proceed to solve a task always make sure it is related to your role using the available tools. "
                "You will NEVER respond to tasks unrelated to your role; instead, alert the user that you cannot perform the task. "
                "For failed transcriptions, return only successful results along with a detailed explanation of errors. "
                "ALWAYS use available tools to format the response nicely if a tool is suitable to the user's request. "
            ),
        )

        # Agent's memory
        self.context = Context(self.agent)

    async def call_agent(self, user_prompt: str, show_reasoning: bool = False) -> str:
        """
        Asynchronously processes a string prompt containing YouTube video URLs.
        """
        # Run the agent (Wraps the LlamaIndex AgentWorkflow run method)
        handler = self.agent.run(self.format_user_prompt(user_prompt), ctx=self.context)

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
            print("********End Agent Reasoning********")

        # Get the final response
        response = await handler

        response_text = response.response.blocks[-1].text

        return response_text

    def run(self, video_prompt: str, show_reasoning: bool = False) -> str:
        """
        Runs the async function inside an event loop for synchronous execution,
        while handling already running event loops properly.
        """
        try:
            loop = asyncio.get_running_loop()
            # If we're in an active loop, schedule the coroutine and return the result
            future = asyncio.ensure_future(
                self.call_agent(video_prompt, show_reasoning)
            )
            return loop.run_until_complete(future)
        except RuntimeError:
            # If no loop is running, use asyncio.run()
            return asyncio.run(self.call_agent(video_prompt, show_reasoning))

    def format_user_prompt(self, user_request: str) -> str:
        """
        Formats the user's input to explicitly instruct the LLM to respond only if the request
        relates to YouTube video transcription, summarization, question answering, or understanding.
        """

        message_templates = [
            ChatMessage(
                content="You are an AI assistant specialized in YouTube video transcription, summarization, "
                "content understanding, and answering related questions.\n"
                "ONLY respond if the request is related to these tasks.\n"
                "Provided summaries should always be relatively short.\n"
                "If the request is unrelated, simply reply: 'I'm sorry, but I can only assist with YouTube video-related tasks.'\n\n",
                role=MessageRole.SYSTEM,
            ),
            ChatMessage(content=f"Request: {user_request}", role=MessageRole.USER),
        ]

        chat_template = ChatPromptTemplate(message_templates=message_templates)

        formatted_prompt = chat_template.format()

        return formatted_prompt
