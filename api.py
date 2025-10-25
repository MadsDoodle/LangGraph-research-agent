from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import json
import asyncio
from datetime import datetime
import uuid

from ai_researcher import INITIAL_PROMPT, graph, start_new_research_session
from langchain_core.messages import AIMessage, HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Research AI Agent API",
    description="Backend API for AI-powered research assistant",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis/database in production)
sessions: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class SessionCreate(BaseModel):
    """Request model for creating a new session"""
    pass

class SessionResponse(BaseModel):
    """Response model for session creation"""
    session_id: str
    thread_id: str
    created_at: str
    message: str

class ChatMessage(BaseModel):
    """Model for chat messages"""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Request model for chat messages"""
    session_id: str
    message: str

class ChatResponse(BaseModel):
    """Response model for chat messages"""
    response: str
    tools_used: List[str]
    papers_found: List[str]
    pdf_path: Optional[str]

class SessionStatus(BaseModel):
    """Model for session status"""
    session_id: str
    chat_history: List[ChatMessage]
    tools_used: List[str]
    papers_found: List[str]
    pdf_path: Optional[str]
    created_at: str

class QuickAction(BaseModel):
    """Model for quick actions"""
    session_id: str
    action: str  # 'write_paper' or 'search_all'


# Helper functions
def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve a session or raise 404"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

def create_session_data() -> Dict[str, Any]:
    """Create new session data structure"""
    config = start_new_research_session()
    return {
        "config": config,
        "chat_history": [],
        "tools_used": set(),
        "papers_found": [],
        "pdf_path": None,
        "created_at": datetime.now().isoformat()
    }


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Research AI Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /sessions": "Create new research session",
            "GET /sessions/{session_id}": "Get session status",
            "POST /chat": "Send chat message",
            "POST /chat/stream": "Stream chat response",
            "POST /quick-action": "Execute quick action",
            "GET /pdf/{session_id}": "Download generated PDF",
            "DELETE /sessions/{session_id}": "Delete session"
        }
    }

@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new research session"""
    try:
        session_id = str(uuid.uuid4())
        session_data = create_session_data()
        sessions[session_id] = session_data
        
        thread_id = session_data["config"]["configurable"]["thread_id"]
        
        logger.info(f"Created new session: {session_id}")
        
        return SessionResponse(
            session_id=session_id,
            thread_id=thread_id,
            created_at=session_data["created_at"],
            message="Session created successfully"
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Get current session status"""
    try:
        session = get_session(session_id)
        
        return SessionStatus(
            session_id=session_id,
            chat_history=[
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in session["chat_history"]
            ],
            tools_used=list(session["tools_used"]),
            papers_found=session["papers_found"],
            pdf_path=session["pdf_path"],
            created_at=session["created_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        del sessions[session_id]
        logger.info(f"Deleted session: {session_id}")
        
        return {"message": "Session deleted successfully", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a chat message and get response"""
    try:
        session = get_session(request.session_id)
        
        # Add user message to history
        session["chat_history"].append({
            "role": "user",
            "content": request.message
        })
        
        # Prepare messages for agent
        messages = [HumanMessage(content=INITIAL_PROMPT)]
        for msg in session["chat_history"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        chat_input = {"messages": messages}
        
        # Process through agent
        full_response = ""
        new_tools = []
        new_papers = []
        
        logger.info(f"Processing chat for session: {request.session_id}")
        
        for s in graph.stream(chat_input, session["config"], stream_mode="values"):
            message = s["messages"][-1]
            
            # Handle tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    session["tools_used"].add(tool_name)
                    new_tools.append(tool_name)
                    logger.info(f"Tool used: {tool_name}")
            
            # Handle AI response
            if isinstance(message, AIMessage) and message.content:
                full_response = message.content if isinstance(message.content, str) else str(message.content)
                
                # Extract paper titles
                import re
                if any(keyword in full_response.lower() for keyword in ['paper', 'title', 'found']):
                    papers = re.findall(r'\"([^\"]+)\"', full_response)
                    new_papers.extend(papers)
                    session["papers_found"].extend(papers)
                
                # Check for PDF generation
                if 'paper_' in full_response and '.pdf' in full_response:
                    pdf_match = re.search(r'(output/paper_\d+_\d+\.pdf)', full_response)
                    if pdf_match:
                        session["pdf_path"] = pdf_match.group(1)
                        logger.info(f"PDF generated: {session['pdf_path']}")
        
        # Add assistant response to history
        session["chat_history"].append({
            "role": "assistant",
            "content": full_response
        })
        
        return ChatResponse(
            response=full_response,
            tools_used=new_tools,
            papers_found=new_papers,
            pdf_path=session["pdf_path"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response (Server-Sent Events)"""
    try:
        session = get_session(request.session_id)
        
        # Add user message to history
        session["chat_history"].append({
            "role": "user",
            "content": request.message
        })
        
        async def event_generator():
            try:
                # Prepare messages for agent
                messages = [HumanMessage(content=INITIAL_PROMPT)]
                for msg in session["chat_history"]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
                
                chat_input = {"messages": messages}
                full_response = ""
                
                for s in graph.stream(chat_input, session["config"], stream_mode="values"):
                    message = s["messages"][-1]
                    
                    # Handle tool calls
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            session["tools_used"].add(tool_name)
                            
                            event_data = {
                                "type": "tool",
                                "tool_name": tool_name
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                    
                    # Handle AI response
                    if isinstance(message, AIMessage) and message.content:
                        content = message.content if isinstance(message.content, str) else str(message.content)
                        full_response = content
                        
                        event_data = {
                            "type": "message",
                            "content": content
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                        
                        # Check for PDF
                        if 'paper_' in content and '.pdf' in content:
                            import re
                            pdf_match = re.search(r'(output/paper_\d+_\d+\.pdf)', content)
                            if pdf_match:
                                session["pdf_path"] = pdf_match.group(1)
                                event_data = {
                                    "type": "pdf",
                                    "pdf_path": session["pdf_path"]
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Add to history
                session["chat_history"].append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # Send completion event
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quick-action", response_model=ChatResponse)
async def quick_action(request: QuickAction):
    """Execute a quick action"""
    try:
        action_messages = {
            "write_paper": "Please write the complete research paper now and generate the PDF.",
            "search_all": "Search all databases for papers related to our topic."
        }
        
        if request.action not in action_messages:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {list(action_messages.keys())}"
            )
        
        # Execute as regular chat
        chat_request = ChatRequest(
            session_id=request.session_id,
            message=action_messages[request.action]
        )
        
        return await chat(chat_request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf/{session_id}")
async def download_pdf(session_id: str):
    """Download the generated PDF for a session"""
    try:
        session = get_session(session_id)
        
        if not session["pdf_path"]:
            raise HTTPException(status_code=404, detail="No PDF generated for this session")
        
        pdf_path = Path(session["pdf_path"])
        
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=pdf_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
