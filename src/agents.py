from dotenv import load_dotenv
from crewai import Agent, LLM

load_dotenv()

llm = LLM(model="gpt-4o-mini")

analyzer_agent = Agent(
    role="Analyzer Agent",
    goal="Analyze papers and documentation to extract actionable insights.",
    backstory="An expert in academic analysis and technical documentation.",
    llm=llm,
    allow_code_execution=True
)

coder_agent = Agent(
    role="Python Developer",
    goal="Generate Python code based on the insights provided by the Analyzer.",
    backstory="Expert in Python development, FastAPI, and SciPy.",
    llm=llm,
    allow_code_execution=True
)


