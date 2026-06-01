import os

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool


llm = LLM(
    model="openhands/gpt-5-2025-08-07",
    api_key=os.getenv("LLM_API_KEY"),
)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ],
)

cwd = os.getcwd()
conversation = Conversation(agent=agent, workspace=cwd)

conversation.send_message("Hi, i wrote a framework called vortex that can enable an abstraction on implementation of sparse attention in sglang, which was very hard to do earlier. Could you based on the example file in this directory: vortex_torch/flow/algorithms.py, propose a better algorithm of dynamic sparse attention? Then, add your algorithm registered name in vortex_torch/examples/verify_algo.sh as the first one.")
conversation.run()
print("All done!")