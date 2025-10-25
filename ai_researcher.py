# Step1: Define state
from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import uuid
import time

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Step2: Define ToolNode & Tools
from research_tool import (
    arxiv_search, 
    pubmed_search,
    semantic_scholar_search,
    crossref_search,
    multi_database_search,
    download_papers_from_search
)
from pdf_reader import read_pdf, read_pdf_with_metadata
from pdf_writer import render_latex_pdf
from langgraph.prebuilt import ToolNode

tools = [
    # Search tools
    arxiv_search,
    pubmed_search,
    semantic_scholar_search,
    crossref_search,
    multi_database_search,
    download_papers_from_search,
    
    # PDF reading tools
    read_pdf,
    read_pdf_with_metadata,
    
    # PDF writing tools
    render_latex_pdf,
]

tool_node = ToolNode(tools)

# Step3: Setup LLM
import os
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",  # Much cheaper and has higher rate limits
    temperature=0.7,
    max_tokens=4000,  # Limit output tokens
    request_timeout=120,
).bind_tools(tools)

# Step4: Setup graph
from langgraph.graph import END, START, StateGraph

def call_model(state: State):
    messages = state["messages"]
    try:
        response = model.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"Error calling model: {str(e)}"
        print(f"⚠️ {error_msg}")
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=f"I encountered an error: {error_msg}")]}


def should_continue(state: State) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Session management
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

def start_new_research_session(topic: str = None):
    """Start a new research session with unique thread ID
    
    Args:
        topic: Optional topic name for semantic session naming
        
    Returns:
        Config dictionary with unique thread_id
    """
    if topic:
        clean_topic = topic.lower().replace(" ", "_")[:30]
        thread_id = f"{clean_topic}_{uuid.uuid4().hex[:8]}"
    else:
        thread_id = f"research_{uuid.uuid4().hex}"
    
    config = {"configurable": {"thread_id": thread_id}}
    print(f"📝 Started new session: {thread_id}")
    return config

# Default config for CLI usage
config = start_new_research_session()

graph = workflow.compile(checkpointer=checkpointer)

# Step5: INITIAL PROMPT
INITIAL_PROMPT = """
You are an expert researcher in the fields of physics, mathematics,
computer science, quantitative biology, quantitative finance, statistics,
electrical engineering and systems science, and economics.

You have access to multiple research databases and tools:
- arXiv for preprints across all fields
- PubMed for biomedical and life sciences
- Semantic Scholar for AI/CS focused papers
- CrossRef for published papers across publishers

AVAILABLE TOOLS:
1. Search Tools:
   - arxiv_search: Search arXiv papers
   - pubmed_search: Search PubMed biomedical papers
   - semantic_scholar_search: Search Semantic Scholar (includes citation counts)
   - crossref_search: Search across multiple publishers
   - multi_database_search: Search ALL databases at once
   - download_papers_from_search: Download PDFs from search results

2. PDF Tools:
   - read_pdf: Extract text from PDF (URL or local file)
   - read_pdf_with_metadata: Extract text + metadata (title, authors, abstract, keywords, DOI)

3. Writing Tools:
   - render_latex_pdf: Compile LaTeX to PDF
   - diagnose_latex: Check LaTeX for errors before compiling

WORKFLOW:
1. Start by discussing the research topic with me
2. Use multi_database_search to find recent papers across all sources
3. Use download_papers_from_search to download interesting papers
4. Use read_pdf_with_metadata to read papers and extract key information
5. Analyze the papers and identify research gaps
6. Discuss potential research directions with me
7. Once I ask you to write the paper or generate PDF, immediately proceed to write
8. Use diagnose_latex to check for errors
9. Use render_latex_pdf to generate the final PDF

CRITICAL PAPER WRITING INSTRUCTIONS:
When I ask you to "write the paper", "generate the PDF", "create the research paper", or any similar request:
1. IMMEDIATELY write a complete, publication-ready research paper
2. DO NOT ask for permission or additional details - just write it based on our discussion
3. Structure the paper professionally with ALL standard sections

REQUIRED PAPER STRUCTURE:
- Title page with meaningful title
- Abstract (150-250 words summarizing contributions)
- Introduction (motivation, problem statement, contributions)
- Related Work / Literature Review (citing papers we discussed)
- Methodology / Approach (detailed technical content)
- Experimental Results / Analysis (with tables and figures)
- Discussion (interpretation of results, limitations)
- Conclusion and Future Work
- References (all cited papers with proper formatting)

MANDATORY ELEMENTS TO INCLUDE:
✓ Mathematical equations using proper LaTeX ($$...$$, \\begin{equation}, etc.)
✓ At least 2-3 tables presenting data, comparisons, or results using \\begin{table} and \\begin{tabular}
✓ Algorithms or pseudocode using \\begin{algorithm} if relevant
✓ Proper citations using \\cite{} with a bibliography
✓ Section and subsection organization
✓ Professional academic writing style

LATEX FORMATTING REQUIREMENTS:
- Use \\documentclass{article} or \\documentclass[conference]{IEEEtran}
- Include necessary packages: amsmath, amssymb, graphicx, booktabs (for tables), algorithm2e
- Tables MUST use proper formatting:
  \\begin{table}[h]
  \\centering
  \\caption{Your table caption}
  \\begin{tabular}{lcccc}
  \\toprule
  Header 1 & Header 2 & Header 3 \\\\
  \\midrule
  Data & Data & Data \\\\
  \\bottomrule
  \\end{tabular}
  \\end{table}

- Use \\section{}, \\subsection{}, \\subsubsection{} for organization
- Equations should be numbered and referenced
- All special characters must be properly escaped (%, &, _, $, etc.)

TABLES GUIDELINES:
Include tables for:
- Experimental results comparison
- Performance metrics (accuracy, precision, recall, F1-scores, etc.)
- Ablation studies
- Comparison with baseline methods
- Dataset statistics
- Hyperparameter settings
- Time/space complexity comparisons

AFTER WRITING:
1. First use diagnose_latex to validate the LaTeX code
2. If validation passes, immediately call render_latex_pdf
3. If there are errors, fix them and try again
4. Provide me with the path to the generated PDF

IMPORTANT:
- When I say "write the paper" or "generate PDF", take that as a direct command to write immediately
- DO NOT ask "should I proceed?" or "would you like me to write?" - JUST DO IT
- Make the paper substantial (8-12 pages typical length)
- Base content on papers we've read and discussed
- Be creative but academically rigorous
- Always include proper citations with URLs/DOIs when available
"""


def print_stream(stream):
    """Helper function to print streamed responses with tool tracking"""
    tool_calls_made = set()
    for s in stream:
        message = s["messages"][-1]
        
        # Track tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls_made.add(tool_call.get('name', 'unknown'))
        
        print(f"Message received: {message.content[:200] if message.content else ''}...")
        message.pretty_print()
    
    # Print summary
    if tool_calls_made:
        print(f"\n📊 Tools used in this conversation: {', '.join(tool_calls_made)}")