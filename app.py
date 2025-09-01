import streamlit as st

st.set_page_config(
    page_title="Legal Document Summarizer",
    page_icon="⚖️",
    layout="wide"
)
import tempfile
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# LangChain imports - Updated versions
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import BaseOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import json
import re
from pydantic import BaseModel, Field
from typing import Optional

class Party(BaseModel):
    name: str = Field(description="Name of the party")
    role: str = Field(description="Role or designation of the party")

class KeyDate(BaseModel):
    type: str = Field(description="Type of date (e.g., effective_date, expiration_date)")
    date: str = Field(description="The actual date")
    description: Optional[str] = Field(description="Additional context about the date")

class Obligation(BaseModel):
    party: str = Field(description="Which party has this obligation")
    obligation: str = Field(description="Description of the obligation")
    deadline: Optional[str] = Field(description="Any deadline associated with this obligation")

class ContractAnalysis(BaseModel):
    summary: str = Field(description="2-3 sentence high-level summary of the contract")
    parties: List[Party] = Field(description="All parties involved in the contract")
    key_dates: List[KeyDate] = Field(description="Important dates in the contract")
    obligations: List[Obligation] = Field(description="Key obligations and responsibilities")

class LegalOutputParser(BaseOutputParser):
    """Custom output parser for legal document analysis"""
    
    def parse(self, text: str) -> ContractAnalysis:
        try:
            #extract JSON from the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return ContractAnalysis(**data)
            else:
                #Fallback parsing if JSON extraction fails
                return self._fallback_parse(text)
        except Exception as e:
            st.error(f"Parsing error: {e}")
            return self._create_empty_analysis()
    
    def _fallback_parse(self, text: str) -> ContractAnalysis:
        """Fallback parsing method"""
        return ContractAnalysis(
            summary="Unable to parse full analysis. Please try uploading the document again.",
            parties=[],
            key_dates=[],
            obligations=[]
        )
    
    def _create_empty_analysis(self) -> ContractAnalysis:
        """Create empty analysis structure"""
        return ContractAnalysis(
            summary="Analysis could not be completed.",
            parties=[],
            key_dates=[],
            obligations=[]
        )

class DocumentProcessor:
    """Handles PDF processing and chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for legal context
            chunk_overlap=300,  # More overlap to preserve context
            separators=[
                "\n\n\n",  
                "\n\n",   
                "\n",      
                ". ",      
                " ",       
                ""
            ],
            keep_separator=True  
        )
    
    def process_pdf(self, file_path: str):
        """Load and split PDF document with better page handling"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Debug: Show what was loaded
            st.info(f"Loaded {len(documents)} pages from PDF")
            
            # Process each page with metadata
            processed_docs = []
            for i, doc in enumerate(documents):
                #Add pg metadata
                doc.metadata['page_number'] = i + 1
                doc.metadata['source_file'] = file_path
                
                #clean up
                cleaned_text = self.clean_text(doc.page_content)
                doc.page_content = cleaned_text
                
                processed_docs.append(doc)
            
            #split doc to chunks
            chunks = self.text_splitter.split_documents(processed_docs)
            
            #chunks added to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
            
            st.info(f"Created {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        #Remove unnecessary  whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        #removing headers n footers
        text = re.sub(r'Page \d+', '', text)
        
        return text.strip()

class VectorStoreManager:
    """Manages ChromaDB vector store"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
    
    def create_vector_store(self, documents):
        """Create vector store from documents"""
        try:
            #vectore store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./chroma_db",
                collection_name="legal_docs"
            )
            
            st.success(f"Vector store created with {len(documents)} document chunks")
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False
    
    def get_retriever(self, k: int = 6):  #retrieve chunks for context
        """Get retriever for RAG"""
        if self.vector_store:
            return self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        return None

class LegalAnalysisChain:
    """Main analysis chain using LCEL"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.output_parser = LegalOutputParser()
        self.chat_history = []  #conversation storing list
        self._setup_prompts()
        self._setup_chain()
    
    def _setup_prompts(self):
        """Setup prompt templates"""
        self.analysis_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a legal document analysis expert. Analyze any type of legal document and extract key information regardless of structure or format.

Document Content:
{context}

Task: {question}

Instructions:
- This could be any type of legal document: contracts, agreements, leases, terms of service, partnership agreements, employment contracts, NDAs, etc.
- The document may have different structures, formatting, and organization
- Adapt your analysis to the specific document type and content provided
- Extract information that exists, don't force information that isn't present

Respond in the following JSON format:
{{
    "summary": "A comprehensive summary explaining what this legal document is about, its main purpose, document type, and key scope/objectives",
    "parties": [
        {{"name": "Party/entity name as written", "role": "Their role, title, or relationship (e.g., Employer, Contractor, Landlord, Licensee, etc.)"}}
    ],
    "key_dates": [
        {{"type": "date_type", "date": "date as written in document", "description": "context or significance of this date"}}
    ],
    "obligations": [
        {{"party": "Party responsible", "obligation": "Description of what they must do/provide/comply with", "deadline": "timeframe if specified"}}
    ]
}}

Analysis Guidelines:
- Identify document type first (service agreement, employment contract, lease, etc.)
- Extract ALL parties regardless of how they're named or referenced
- Find dates of any significance: effective dates, deadlines, renewal dates, notice periods, payment dates, etc.
- Capture ALL types of obligations: financial, performance, compliance, reporting, confidentiality, etc.
- Handle blank fields appropriately - note if information is template/unfilled
- Work with any document structure - sections, clauses, paragraphs, bullet points, etc.
- Be flexible with terminology - parties might be called clients, vendors, employees, tenants, etc.

Respond ONLY with valid JSON in the exact format above.
"""
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""
You are a legal document expert answering questions about any type of legal document.

Document Context:
{context}

Previous conversation:
{chat_history}

Question: {question}

Instructions:
- This could be any type of legal document with any structure
- Review all available content to answer the question
- Adapt your response to the specific document type and question asked
- Provide specific references when possible (section numbers, clauses, page references, etc.)
- If the question asks about something not in the document, clearly state that

Provide a clear, detailed answer based on the document content:
"""
        )
    
    def _setup_chain(self):
        """Setup LCEL chain composition"""
        #Main analysis chain
        self.analysis_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.analysis_prompt
            | self.llm
            | self.output_parser
        )
        
        #Q&A chain
        self.qa_chain = (
            {
                "context": self.retriever,
                "chat_history": lambda x: self._format_chat_history(),
                "question": RunnablePassthrough()
            }
            | self.qa_prompt
            | self.llm
        )
    
    def _format_chat_history(self):
        """Format chat history for prompt"""
        if not self.chat_history:
            return "No previous conversation."
        
        formatted = ""
        for i in range(len(self.chat_history)):
            if i % 2 == 0:  # User message
                formatted += f"User: {self.chat_history[i]}\n"
            else:  # AI message
                formatted += f"Assistant: {self.chat_history[i]}\n"
        return formatted[-1000:]  # Keep last 1000 chars
    
    def analyze_contract(self) -> ContractAnalysis:
        """Perform comprehensive legal document analysis"""
        try:
            question = "Analyze this legal document comprehensively. Identify the document type and extract all parties, key dates, and obligations present in the document."
            result = self.analysis_chain.invoke(question)
            return result
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return self.output_parser._create_empty_analysis()
    
    def ask_question(self, question: str) -> str:
        """Ask follow-up question about the document"""
        try:
            response = self.qa_chain.invoke(question)
            # Add to simple chat history
            self.chat_history.append(question)
            self.chat_history.append(response)
            # Keep only last 10 exchanges
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            return response
        except Exception as e:
            return f"Sorry, I couldn't process your question: {e}"

# streamlit

def init_session_state():
    """Initialize session state variables"""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'current_document' not in st.session_state:
        st.session_state.current_document = None
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    # Initialize memory for chat history
    if 'memory' not in st.session_state:
        st.session_state.memory = []

def setup_ollama_components():
    """Initialize Ollama LLM and embeddings"""
    try:
        llm = OllamaLLM(
            model="llama3.1:8b",
            temperature=0.1,
            num_ctx=4096
        )
        
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        
        
        test_response = llm.invoke("Hello")
        return llm, embeddings
    except Exception as e:
        st.error(f"Ollama connection error: {e}")
        st.error("Please ensure Ollama is running with required models.")
        st.info("Run: `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`")
        return None, None

def display_analysis_results(analysis: ContractAnalysis):
    """Display analysis results in readable format for any legal document"""
    
    st.markdown("## 📄 Legal Document Analysis")
    
    #summary
    st.markdown("### 📋 Document Summary")
    st.write(analysis.summary)
    
    #parties
    st.markdown("### 👥 Parties Involved")
    if analysis.parties:
        for party in analysis.parties:
            st.write(f"• **{party.name}** - {party.role}")
    else:
        st.write("No parties information found in this document.")
    
    #imp dates
    st.markdown("### 📅 Important Dates")
    if analysis.key_dates:
        for date in analysis.key_dates:
            desc = f" - {date.description}" if date.description else ""
            st.write(f"• **{date.type.replace('_', ' ').title()}**: {date.date}{desc}")
    else:
        st.write("No specific dates found in this document.")
    
    #Obligations Section
    st.markdown("### ⚖️ Key Obligations & Responsibilities")
    if analysis.obligations:
        for obligation in analysis.obligations:
            deadline = f" (Deadline: {obligation.deadline})" if obligation.deadline else ""
            st.write(f"• **{obligation.party}**: {obligation.obligation}{deadline}")
    else:
        st.write("No specific obligations extracted from this document.")

def main():
    st.title("⚖️ Legal Document Summarizer")
    st.markdown("Upload any legal document (contracts, agreements, leases, etc.) to get an AI-powered analysis and extract key information.")
    
    # Initialize session state
    init_session_state()
    
    #ollama setup
    llm, embeddings = setup_ollama_components()
    if not llm or not embeddings:
        st.stop()
    
    #fileupload
    st.markdown("### 📎 Upload Legal Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any legal document: contracts, agreements, leases, terms of service, NDAs, etc."
    )
    
    if uploaded_file is not None:
        # Check if this is a new document
        if st.session_state.current_document != uploaded_file.name:
            st.session_state.current_document = uploaded_file.name
            st.session_state.analysis_result = None
            st.session_state.chain = None
            # Reset memory for new document
            st.session_state.memory = []
            
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Process document
                    processor = DocumentProcessor()
                    chunks = processor.process_pdf(tmp_file_path)
                    
                    if chunks:
                        # Create vector store
                        vector_manager = VectorStoreManager(embeddings)
                        if vector_manager.create_vector_store(chunks):
                            retriever = vector_manager.get_retriever()
                            
                            # Create analysis chain
                            st.session_state.chain = LegalAnalysisChain(
                                llm=llm,
                                retriever=retriever
                            )
                            
                            # Perform analysis
                            with st.spinner("Analyzing contract..."):
                                st.session_state.analysis_result = st.session_state.chain.analyze_contract()
                        
                finally:
                    # Cleanup temp file
                    os.unlink(tmp_file_path)
    
    # Display results
    if st.session_state.analysis_result:
        display_analysis_results(st.session_state.analysis_result)
        
        # Q&A Section
        st.markdown("---")
        st.markdown("### 💬 Ask Questions About This Document")
        
        question = st.text_input(
            "Ask any question about the legal document:",
            placeholder="e.g., What are the payment terms? Who are the parties? What are the termination conditions?"
        )
        
        if question and st.session_state.chain:
            with st.spinner("Finding answer..."):
                answer = st.session_state.chain.ask_question(question)
                st.markdown("**Answer:**")
                st.write(answer)
#installing instructions
installation_instructions = """
# Legal Document Summarizer Setup Instructions

## 1. Install Ollama
Visit: https://ollama.ai and download Ollama for your OS

## 2. Pull Required Models
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 4. Run the Application
```bash
streamlit run app.py
```

## 5. Usage
1. Upload a PDF legal contract
2. Wait for automatic analysis
3. Review the summary, parties, dates, and obligations
4. Ask follow-up questions about the contract

## Troubleshooting
- Ensure Ollama is running: `ollama serve`
- Check models are installed: `ollama list`
- For large documents, processing may take 1-2 minutes
"""

if __name__ == "__main__":
    # Display setup instructions in sidebar
    with st.sidebar:
        st.markdown("### 🚀 Setup Instructions")
        with st.expander("Click to view setup steps"):
            st.markdown(installation_instructions)
        
        st.markdown("### ℹ️ About")
        st.write("This app uses:")
        st.write("• LangChain for document processing")
        st.write("• Ollama for local LLM inference")
        st.write("• ChromaDB for vector storage")
        st.write("• RAG for intelligent retrieval")
    
    main()

