class ToolPairPreservationHook(HookProvider):
    """Use constant argument values for specific parameters of a tool."""

    def __init__(self):
        """
        Initialize fixed parameter values for tools.

        Args:
            fixed_tool_arguments: A dictionary mapping tool names to dictionaries of
                parameter names and their fixed values. These values will override any
                values provided by the agent when the tool is invoked.
        """
        pass

    def check_tool_consistency(self, event):
        agent_messages = event.agent.messages
        # print(agent_messages)
        if not agent_messages:
            return

        found_orphan_tool_use = False
        if agent_messages[-1]["role"] == "assistant":
            for content in agent_messages[-1]["content"]:
                if "toolUse" in content:
                    print("FOUND TOOL USE AS LAST MESSAGE")
                    found_orphan_tool_use = True

        if found_orphan_tool_use:
            agent_messages.pop(-1)
            print("REMOVED TOOL USE AS LAST MESSAGE")
            # print(agent_messages)
            event.agent.messages = agent_messages

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self.check_tool_consistency)