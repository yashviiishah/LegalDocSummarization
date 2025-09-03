
# Legal Document Summarizer

A powerful AI-powered application that analyzes legal documents and extracts key information using LangChain, Ollama, and Streamlit. Upload any PDF legal document to get comprehensive analysis including parties, key dates, obligations, and interactive Q&A capabilities.

## Project Demonstration
[https://youtu.be/qyIP7CT8FUw](url)


  [![Watch the video](https://img.youtube.com/vi/qyIP7CT8FUw/0.jpg)](https://youtu.be/qyIP7CT8FUw)

## Features

- **Universal Legal Document Support**: Handles contracts, agreements, leases, terms of service, NDAs, employment contracts, and more
- **Comprehensive Analysis**: Automatically extracts parties, key dates, obligations, and document summaries
- **Interactive Q&A**: Ask follow-up questions about any aspect of the uploaded document
- **Local AI Processing**: Uses Ollama for privacy-focused, offline document analysis
- **Smart Chunking**: Optimized text processing for legal document structures
- **Vector Search**: RAG-powered retrieval for accurate question answering


### Document Upload & Analysis
The application provides a clean interface for uploading PDF documents and displays structured analysis results.

### Interactive Q&A
Ask natural language questions about the document and get contextual answers based on the document content.

## Prerequisites

- Python 3.8+
- Ollama installed and running
- Required Ollama models (automatically prompted for installation)

## Quick Start

### 1. Install Ollama
Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Clone Repository
```bash
git clone https://github.com/yashviiishah/LegalDocSummarization.git
cd legal-document-summarizer
```

### 3. Run Setup Script

## Manual Installation

### 1. Install Required Models
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```

## Usage

1. **Start the Application**: Run `streamlit run app.py`
2. **Upload Document**: Click "Choose a PDF file" and select your legal document
3. **Wait for Processing**: The app will automatically process and analyze the document
4. **Review Analysis**: See extracted parties, dates, obligations, and summary
5. **Ask Questions**: Use the Q&A section to ask specific questions about the document

## Supported Document Types

- Service Agreements
- Employment Contracts
- Lease Agreements
- Non-Disclosure Agreements (NDAs)
- Partnership Agreements
- Terms of Service
- Privacy Policies
- Purchase Agreements
- Licensing Agreements


## Tech Stack:

- **Frontend**: Streamlit for web interface
- **LLM**: Ollama with Llama 3.1 8B model
- **Embeddings**: Nomic Embed Text model
- **Vector Database**: ChromaDB for document storage and retrieval
- **Framework**: LangChain for document processing and RAG implementation
- **PDF Processing**: PyPDF2 for document loading
- **Structured Output**: Pydantic for data validation

## Architecture

```
PDF Upload → Document Processing → Text Chunking → Vector Embeddings → ChromaDB Storage
                                                                              ↓
User Questions ← RAG Retrieval ← LLM Analysis ← Structured Output Parser ← Vector Search
```

## Configuration

The application uses sensible defaults but can be customized:

- **Chunk Size**: 1500 characters (optimized for legal context)
- **Chunk Overlap**: 300 characters (preserves context across chunks)
- **Retrieval Count**: 6 most relevant chunks per query
- **Temperature**: 0.1 (focused, deterministic responses)

## Troubleshooting

### Common Issues

**"Ollama connection error"**
- Ensure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`
- Verify models are pulled: `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`

**"Error processing PDF"**
- Ensure the PDF is not password-protected
- Check that the PDF contains readable text (not just images)
- Try with a smaller document first

**Slow processing**
- Large documents may take 1-2 minutes to process
- Consider using a more powerful machine for faster processing
- Ensure adequate RAM (4GB+ recommended)

### Performance Tips

- Use PDFs with clear, searchable text for best results
- Smaller documents (under 50 pages) process faster
- Close other resource-intensive applications during processing

## Development

### Project Structure
```
legal-document-summarizer/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── README.md             # This file
└── chroma_db/            # Vector database storage (created automatically)
```

### Key Components

- **DocumentProcessor**: Handles PDF loading and text chunking
- **VectorStoreManager**: Manages ChromaDB vector storage
- **LegalAnalysisChain**: Main analysis logic using LangChain LCEL
- **LegalOutputParser**: Converts LLM output to structured format

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## Security & Privacy

- **Local Processing**: All document analysis happens locally on your machine
- **No Data Transmission**: Documents are not sent to external servers
- **Temporary Storage**: Uploaded files are processed in memory and temporary files are cleaned up
- **Privacy First**: Your legal documents remain completely private

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for informational purposes only and does not constitute legal advice. Always consult with qualified legal professionals for legal matters. The accuracy of AI-generated analysis may vary depending on document complexity and format.

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Ensure all prerequisites are met
3. Open an issue on GitHub with error details and document type
4. Include system information (OS, Python version, Ollama version)

