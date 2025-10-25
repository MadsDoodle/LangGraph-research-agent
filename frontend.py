import streamlit as st
from ai_researcher import INITIAL_PROMPT, graph, start_new_research_session
from pathlib import Path
import logging
from langchain_core.messages import AIMessage, HumanMessage
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced app config
st.set_page_config(
    page_title="Research AI Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .tool-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        background-color: #e3f2fd;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        color: #1976d2;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.config = start_new_research_session()
    st.session_state.pdf_path = None
    st.session_state.tools_used = set()
    st.session_state.papers_found = []
    st.session_state.session_started = False
    logger.info("Initialized new session")

# Sidebar
with st.sidebar:
    st.header("🔬 Research Session")
    
    # Session info
    if st.session_state.config:
        thread_id = st.session_state.config["configurable"]["thread_id"]
        st.info(f"**Session ID:** `{thread_id[:20]}...`")
    
    # New session button
    if st.button("🔄 Start New Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.config = start_new_research_session()
        st.session_state.pdf_path = None
        st.session_state.tools_used = set()
        st.session_state.papers_found = []
        st.session_state.session_started = False
        st.rerun()
    
    st.divider()
    
    # Tools used tracker
    st.subheader("🛠️ Tools Used")
    if st.session_state.tools_used:
        for tool in sorted(st.session_state.tools_used):
            st.markdown(f'<span class="tool-badge">✓ {tool}</span>', unsafe_allow_html=True)
    else:
        st.caption("No tools used yet")
    
    st.divider()
    
    # Papers found tracker
    st.subheader("📚 Papers Found")
    if st.session_state.papers_found:
        st.metric("Total Papers", len(st.session_state.papers_found))
        with st.expander("View Papers"):
            for i, paper in enumerate(st.session_state.papers_found[:10], 1):
                st.caption(f"{i}. {paper[:60]}...")
    else:
        st.caption("No papers found yet")
    
    st.divider()
    
    # Generated PDF viewer
    if st.session_state.pdf_path:
        st.subheader("📄 Generated PDF")
        pdf_path = Path(st.session_state.pdf_path)
        if pdf_path.exists():
            st.success(f"✓ PDF Generated")
            st.caption(f"**Location:** `{pdf_path.name}`")
            
            # Download button
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=f,
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    use_container_width=True
                )
    
    st.divider()
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Write Paper", use_container_width=True):
            st.session_state.quick_action = "write_paper"
    with col2:
        if st.button("🔍 Search All", use_container_width=True):
            st.session_state.quick_action = "search_all"

# Main chat interface
st.header("💬 Chat with Research Agent")

# Welcome message
if not st.session_state.session_started:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Welcome to the Research AI Agent!**
        
        I can help you:
        - 🔍 Search across multiple research databases (arXiv, PubMed, Semantic Scholar, CrossRef)
        - 📥 Download and read research papers
        - 📊 Analyze papers and identify research gaps
        - ✍️ Write comprehensive research papers with tables, equations, and citations
        - 📄 Generate publication-ready PDFs
        
        **Get started by telling me what research topic interests you!**
        """)
    st.session_state.session_started = True

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle quick actions
if "quick_action" in st.session_state and st.session_state.quick_action:
    action = st.session_state.quick_action
    if action == "write_paper":
        user_input = "Please write the complete research paper now and generate the PDF."
    elif action == "search_all":
        user_input = "Search all databases for papers related to our topic."
    st.session_state.quick_action = None
    st.rerun()

# Chat input
user_input = st.chat_input("Ask me anything about research...")

if user_input:
    # Log and display user input
    logger.info(f"User input: {user_input}")
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare input for the agent
    messages = [HumanMessage(content=INITIAL_PROMPT)]
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    chat_input = {"messages": messages}
    logger.info("Starting agent processing...")

    # Create placeholder for streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""
        current_tool = None
        
        # Stream agent response
        try:
            for s in graph.stream(chat_input, st.session_state.config, stream_mode="values"):
                message = s["messages"][-1]
                
                # Handle tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        current_tool = tool_name
                        st.session_state.tools_used.add(tool_name)
                        logger.info(f"Tool call: {tool_name}")
                        
                        # Show tool status
                        status_placeholder.info(f"🔧 Using tool: **{tool_name}**")
                        
                        # Track papers if search tool
                        if 'search' in tool_name.lower():
                            status_placeholder.info(f"🔍 Searching for papers...")
                
                # Handle assistant response
                if isinstance(message, AIMessage) and message.content:
                    text_content = message.content if isinstance(message.content, str) else str(message.content)
                    full_response = text_content
                    message_placeholder.markdown(full_response)
                    
                    # Extract paper titles from response (simple heuristic)
                    if any(keyword in text_content.lower() for keyword in ['paper', 'title', 'found']):
                        # Look for quoted text or numbered lists
                        papers = re.findall(r'\"([^\"]+)\"', text_content)
                        st.session_state.papers_found.extend(papers)
                    
                    # Check if PDF was generated
                    if 'paper_' in text_content and '.pdf' in text_content:
                        pdf_match = re.search(r'(output/paper_\d+_\d+\.pdf)', text_content)
                        if pdf_match:
                            st.session_state.pdf_path = pdf_match.group(1)
                            logger.info(f"PDF generated: {st.session_state.pdf_path}")
            
            # Clear status after completion
            status_placeholder.empty()
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            error_message = f"⚠️ An error occurred: {str(e)}"
            message_placeholder.error(error_message)
            full_response = error_message

    # Add final response to history
    if full_response:
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        
        # Show success message if PDF was generated
        if st.session_state.pdf_path:
            st.balloons()
            st.success("🎉 Research paper generated successfully! Check the sidebar to download.")
    
    # Rerun to update sidebar
    st.rerun()

# Footer
st.divider()
st.caption("🔬 Powered by LangGraph, OpenAI GPT-4, and multiple research databases")