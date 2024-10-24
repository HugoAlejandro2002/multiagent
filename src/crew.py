from crewai import Crew
from agents import (
    analyzer_agent, 
    coder_agent
)
from tasks import (
    analyze_paper_task,
    read_fastapi_docs_task,
    read_scipy_docs_task,
    combine_analysis_task,
    generate_service_task
)

def get_crew():
    return Crew(
        agents=[analyzer_agent, coder_agent],
        tasks=[
            analyze_paper_task,
            read_fastapi_docs_task,
            read_scipy_docs_task,
            combine_analysis_task,
            generate_service_task
        ],
        verbose=True
    )
