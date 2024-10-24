import os
from crewai import Task
from crewai_tools import PDFSearchTool, CodeDocsSearchTool
from agents import (
    analyzer_agent, 
    coder_agent
)

base_dir = os.path.dirname(os.path.abspath(__file__))
papers_path = os.path.join(base_dir, "..", "papers", "test.pdf")
artifacts_dir = os.path.join(base_dir, "..", "artifacts")
analysis_summary_path = os.path.join(artifacts_dir, "analysis_summary.txt")
service_output_path = os.path.join(artifacts_dir, "fastapi_service.py")

pdf_tool = PDFSearchTool(pdf=papers_path)
fastapi_docs_tool = CodeDocsSearchTool(docs_url="https://fastapi.tiangolo.com")
scipy_docs_tool = CodeDocsSearchTool(docs_url="https://docs.scipy.org/doc/scipy/")

analyze_paper_task = Task(
    description="Extract key insights from the academic paper.",
    expected_output="A summary of the key points relevant to the service.",
    agent=analyzer_agent,
    tools=[pdf_tool],
    output_file=analysis_summary_path
)


read_fastapi_docs_task = Task(
    description="Extract relevant examples from FastAPI documentation.",
    expected_output="Examples or snippets relevant to the service from FastAPI.",
    agent=analyzer_agent,
    tools=[fastapi_docs_tool]
)

read_scipy_docs_task = Task(
    description="Extract relevant examples from SciPy documentation.",
    expected_output="Examples or snippets relevant to the service from SciPy.",
    agent=analyzer_agent,
    tools=[scipy_docs_tool]
)

combine_analysis_task = Task(
    description=(
        "Combine the insights from the paper and documentation into a cohesive plan "
        "for the FastAPI service."
    ),
    expected_output="A consolidated analysis with a plan for the service implementation.",
    agent=analyzer_agent
)


generate_service_task = Task(
    description=(
        "Generate a FastAPI service and additional modules based on the combined analysis."
    ),
    expected_output="A FastAPI service and helper modules saved as Python files.",
    agent=coder_agent,
    dependencies=[combine_analysis_task],
    output_file=service_output_path
)
