import logging
from strands import Agent
from strands.multiagent import GraphBuilder
from strands.models import BedrockModel

# Enable debug logs and print them to stderr
logging.getLogger("strands.multiagent").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

def build_graph(model_id, max_tokens):
    model = BedrockModel(
        model_id=model_id,
        max_tokens=max_tokens)

    # Create specialized agents
    researcher = Agent(model=model, name="researcher", system_prompt="You are a research specialist...")
    analyst = Agent(model=model, name="analyst", system_prompt="You are a data analysis specialist...")
    fact_checker = Agent(model=model, name="fact_checker", system_prompt="You are a fact checking specialist...")
    report_writer = Agent(model=model, name="report_writer", system_prompt="You are a report writing specialist...")

    # Build the graph
    builder = GraphBuilder()

    # Add nodes
    builder.add_node(researcher, "research")
    builder.add_node(analyst, "analysis")
    builder.add_node(fact_checker, "fact_check")
    builder.add_node(report_writer, "report")

    # Add edges (dependencies)
    builder.add_edge("research", "analysis")
    builder.add_edge("research", "fact_check")
    builder.add_edge("analysis", "report")
    builder.add_edge("fact_check", "report")

    # Set entry points (optional - will be auto-detected if not specified)
    builder.set_entry_point("research")

    # Build the graph
    graph = builder.build()

    return graph