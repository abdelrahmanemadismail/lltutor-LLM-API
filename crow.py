from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
import wolframalpha

os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model = "crewai-llama2",
    base_url = "http://localhost:11434/v1")

app_id = "XRP95P-AT5K72TQGP"
wolfram_client = wolframalpha.Client(app_id)

def query_wolfram_alpha(query):
    """Query Wolfram Alpha and return the first result."""
    res = wolfram_client.query(query)
    try:
        return next(res.results).text
    except StopIteration:
        return None

general_agent = Agent(
    role="AI Math Tutor",
    goal="""The goal is to format the result in understandable format for students, on markdown math format.""",
    backstory="""You are an AI math tutor specialized in Mathematics. You focus on delivering precise and relevant answers to inquiries directly related to math topics.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)


task = Task(description="""solve the integral of x^3 dx""",
             agent=general_agent,
             expected_output="A step-by-step solution.")

crew = Crew(
            agents=[general_agent],
            tasks=[task],
            verbose=2
        )

wolfram_query_result = query_wolfram_alpha("integrate x^3 dx")
print("Wolfram Alpha Query Result:", wolfram_query_result)

result = crew.kickoff()

print(result)