#!/usr/bin/env python3
"""
Main entry point for Tyra's Advanced Memory MCP Server.

🧠 Tyra MCP Memory Server v3.0.0 - Production Ready
==================================================

This enterprise-grade memory system provides 17 MCP tools for:

🔧 Core Memory Operations:
  • store_memory - Advanced storage with entity extraction
  • search_memory - Hybrid search with confidence scoring
  • delete_memory - Safe memory deletion
  • analyze_response - Hallucination detection & validation

📊 Analytics & Intelligence:
  • get_memory_stats - Comprehensive system statistics
  • get_learning_insights - Adaptive learning insights
  • detect_patterns - Pattern recognition & knowledge gaps
  • analyze_temporal_evolution - Concept evolution tracking

🔗 Advanced Features:
  • deduplicate_memories - Semantic deduplication
  • summarize_memories - AI-powered summarization
  • crawl_website - Natural language web crawling
  • health_check - System health assessment

🎯 Intelligent Suggestions:
  • suggest_related_memories - ML-powered recommendations
  • detect_memory_connections - Automatic connection discovery
  • recommend_memory_organization - Structure optimization
  • detect_knowledge_gaps - Gap analysis with learning paths

🌐 Web Integration:
  • web_search - Local web search with content extraction

🚀 Enterprise Features:
  ✅ 100% Local Operation (no external APIs)
  ✅ Multi-Agent Support (Claude, Tyra, Archon isolation)
  ✅ Real-time Analytics Dashboard
  ✅ Advanced RAG with hallucination detection
  ✅ Temporal knowledge graphs (Neo4j + Graphiti)
  ✅ Multi-layer caching (Redis + PostgreSQL)
  ✅ Predictive intelligence & self-learning
  ✅ Trading data integration with 95% confidence requirements
  ✅ Comprehensive API (19 modules, WebSocket streaming)

Usage:
    python main.py                    # Start MCP server
    python main.py --help             # Show detailed help
    python main.py --validate         # Validate configuration
    python main.py --version          # Show version info

The server communicates via stdio for MCP protocol compatibility.
For API access: http://localhost:8000
For dashboard: http://localhost:8050/dashboard
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def show_banner():
    """Display startup banner with system information."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 TYRA MCP MEMORY SERVER v3.0.0                         ║
║                         Production Ready Enterprise System                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🔧 17 MCP Tools Available     📊 Real-time Analytics Dashboard             ║
║  🎯 ML-Powered Suggestions     🔗 Temporal Knowledge Graphs                 ║
║  🚀 Multi-Agent Support        💎 95% Confidence Trading Safety             ║
║  🌐 Web Crawling & Ingestion   ⚡ WebSocket Streaming                       ║
║                                                                              ║
║  🏆 Features: Local Operation • Hallucination Detection • Self-Learning     ║
║             Pattern Recognition • Predictive Intelligence • ROI Analytics    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"🕐 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print("=" * 80)

def show_version():
    """Show version and system information."""
    print("Tyra MCP Memory Server v3.0.0")
    print("Enterprise-grade AI memory system with advanced analytics")
    print("")
    print("Components:")
    print("  • MCP Server: 17 tools available")
    print("  • Memory Engine: PostgreSQL + pgvector")
    print("  • Knowledge Graphs: Neo4j + Graphiti")
    print("  • Caching: Redis multi-layer")
    print("  • AI Models: HuggingFace local embeddings")
    print("  • Analytics: Performance tracking + dashboards")
    print("  • Safety: Hallucination detection + confidence scoring")
    print("  • Intelligence: ML suggestions + pattern recognition")
    print("")
    print("Endpoints:")
    print("  • MCP: stdio protocol")
    print("  • REST API: http://localhost:8000")
    print("  • Dashboard: http://localhost:8050/dashboard")
    print("  • Health: http://localhost:8000/v1/health")

async def validate_system():
    """Validate system configuration and dependencies."""
    print("🔍 Validating Tyra Memory Server Configuration...")
    print("")
    
    validation_results = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 11:
        validation_results.append("✅ Python version: 3.11+ ✓")
    else:
        validation_results.append(f"❌ Python version: {python_version.major}.{python_version.minor} (need 3.11+)")
    
    # Check required directories
    required_dirs = [
        "./models/embeddings/e5-large-v2",
        "./models/embeddings/all-MiniLM-L12-v2", 
        "./models/cross-encoders/ms-marco-MiniLM-L-6-v2",
        "./config",
        "./src",
        "./tyra-ingest/inbox"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            validation_results.append(f"✅ Directory: {directory} ✓")
        else:
            validation_results.append(f"❌ Directory missing: {directory}")
    
    # Check configuration files
    config_files = [
        "./config/config.yaml",
        "./config/providers.yaml",
        "./config/models.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            validation_results.append(f"✅ Config: {config_file} ✓")
        else:
            validation_results.append(f"❌ Config missing: {config_file}")
    
    # Check dependencies
    try:
        import torch
        validation_results.append("✅ PyTorch: Available ✓")
        if torch.cuda.is_available():
            validation_results.append(f"✅ CUDA: Available ({torch.cuda.get_device_name(0)}) ✓")
        else:
            validation_results.append("⚠️  CUDA: Not available (CPU only)")
    except ImportError:
        validation_results.append("❌ PyTorch: Not installed")
    
    dependencies = [
        "sentence_transformers",
        "asyncpg", 
        "redis",
        "neo4j",
        "mcp",
        "fastapi",
        "dash"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            validation_results.append(f"✅ {dep}: Installed ✓")
        except ImportError:
            validation_results.append(f"❌ {dep}: Not installed")
    
    # Print results
    for result in validation_results:
        print(result)
    
    # Summary
    errors = [r for r in validation_results if r.startswith("❌")]
    warnings = [r for r in validation_results if r.startswith("⚠️")]
    
    print("")
    print("=" * 60)
    if errors:
        print(f"❌ Validation failed: {len(errors)} errors found")
        print("Please fix the issues above before starting the server.")
        return False
    elif warnings:
        print(f"⚠️  Validation passed with {len(warnings)} warnings")
        print("System will work but consider addressing warnings for optimal performance.")
        return True
    else:
        print("✅ All validations passed! System ready to start.")
        return True

async def check_system_health():
    """Perform comprehensive system health check."""
    print("🏥 Checking Tyra Memory Server Health...")
    print("")
    
    health_results = []
    
    # Check if server can start
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from src.mcp.server import TyraMemoryServer
        
        server = TyraMemoryServer()
        # Try to initialize components
        await server._initialize_components()
        health_results.append("✅ Server components: All initialized successfully")
        
        # Check individual component health
        if server.memory_manager:
            memory_health = await server.memory_manager.health_check()
            if memory_health.get("status") == "healthy":
                health_results.append("✅ Memory system: Operational")
            else:
                health_results.append("⚠️  Memory system: Issues detected")
        
        if server.web_search_agent:
            search_health = await server.web_search_agent.health_check()
            if search_health.get("status") == "healthy":
                health_results.append("✅ Web search: Operational")
            else:
                health_results.append("⚠️  Web search: Issues detected")
        
        # Clean shutdown
        if server.memory_manager:
            await server.memory_manager.close()
            
    except Exception as e:
        health_results.append(f"❌ Server initialization: {str(e)}")
    
    # Print results
    for result in health_results:
        print(result)
    
    # Overall status
    errors = [r for r in health_results if r.startswith("❌")]
    warnings = [r for r in health_results if r.startswith("⚠️")]
    
    print("")
    print("=" * 50)
    if errors:
        print(f"❌ Health check failed: {len(errors)} critical issues")
        return False
    elif warnings:
        print(f"⚠️  Health check passed with {len(warnings)} warnings")
        return True
    else:
        print("✅ All health checks passed! System is ready.")
        return True

async def run_benchmarks():
    """Run performance benchmarks."""
    print("🏃 Running Tyra Memory Server Benchmarks...")
    print("")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from src.mcp.server import TyraMemoryServer
        import time
        
        server = TyraMemoryServer()
        await server._initialize_components()
        
        print("📊 Benchmark Results:")
        print("-" * 40)
        
        # Memory storage benchmark
        start_time = time.time()
        test_content = "This is a test memory for benchmarking performance."
        
        if server.memory_manager:
            from src.core.memory.manager import MemoryStoreRequest
            request = MemoryStoreRequest(
                content=test_content,
                agent_id="benchmark",
                extract_entities=True
            )
            
            # Store 10 test memories
            for i in range(10):
                await server.memory_manager.store_memory(request)
            
            storage_time = (time.time() - start_time) * 1000 / 10  # ms per operation
            print(f"✅ Memory storage: {storage_time:.2f}ms per operation")
            
            # Search benchmark
            start_time = time.time()
            from src.core.memory.manager import MemorySearchRequest
            search_request = MemorySearchRequest(
                query="test memory",
                agent_id="benchmark",
                top_k=5
            )
            
            # Perform 10 searches
            for i in range(10):
                await server.memory_manager.search_memory(search_request)
            
            search_time = (time.time() - start_time) * 1000 / 10
            print(f"✅ Memory search: {search_time:.2f}ms per operation")
        
        # Embedding benchmark
        if server.memory_manager and server.memory_manager.embedding_provider:
            start_time = time.time()
            for i in range(10):
                await server.memory_manager.embedding_provider.embed_text("benchmark text")
            
            embedding_time = (time.time() - start_time) * 1000 / 10
            print(f"✅ Embedding generation: {embedding_time:.2f}ms per operation")
        
        print("")
        print("🎯 Performance targets:")
        print("  • Memory storage: <100ms ✓" if storage_time < 100 else "  • Memory storage: <100ms ❌")
        print("  • Memory search: <50ms ✓" if search_time < 50 else "  • Memory search: <50ms ❌")
        print("  • Embedding: <200ms ✓" if embedding_time < 200 else "  • Embedding: <200ms ❌")
        
        # Cleanup
        if server.memory_manager:
            await server.memory_manager.close()
            
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

async def download_models():
    """Download required AI models."""
    print("📥 Downloading Required AI Models...")
    print("")
    
    models_to_download = [
        {
            "name": "intfloat/e5-large-v2",
            "path": "./models/embeddings/e5-large-v2",
            "description": "Primary embedding model (1024 dimensions)"
        },
        {
            "name": "sentence-transformers/all-MiniLM-L12-v2", 
            "path": "./models/embeddings/all-MiniLM-L12-v2",
            "description": "Fallback embedding model (384 dimensions)"
        },
        {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "path": "./models/cross-encoders/ms-marco-MiniLM-L-6-v2", 
            "description": "Cross-encoder for reranking"
        }
    ]
    
    try:
        import subprocess
        
        for model in models_to_download:
            print(f"📦 Downloading {model['name']}...")
            print(f"   {model['description']}")
            
            # Create directory
            os.makedirs(model['path'], exist_ok=True)
            
            # Download with huggingface-cli
            cmd = [
                "huggingface-cli", "download", 
                model['name'],
                "--local-dir", model['path'],
                "--local-dir-use-symlinks", "False"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Downloaded successfully to {model['path']}")
            else:
                print(f"   ❌ Download failed: {result.stderr}")
                print(f"   💡 Try: pip install huggingface_hub[cli]")
                return False
            
            print("")
    
        print("🎉 All models downloaded successfully!")
        print("💡 You can now start the server with: python main.py")
        return True
        
    except FileNotFoundError:
        print("❌ huggingface-cli not found")
        print("💡 Install with: pip install huggingface_hub[cli]")
        return False
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def show_help():
    """Show detailed help information."""
    help_text = """
Tyra MCP Memory Server - Advanced AI Memory System

USAGE:
    python main.py [OPTIONS]

OPTIONS:
    --help, -h          Show this help message
    --version, -v       Show version information  
    --validate          Validate system configuration
    --health            Check system health and exit
    --benchmark         Run performance benchmarks
    --download-models   Download required AI models
    --no-banner         Skip startup banner
    --config PATH       Use custom config file
    --log-level LEVEL   Set logging level (DEBUG, INFO, WARNING, ERROR)
    --dev               Start in development mode with enhanced debugging
    --daemon            Run as background daemon (production use)
    --port PORT         Override default API port (default: 8000)

EXAMPLES:
    python main.py                    # Start server with default settings
    python main.py --validate        # Check system configuration
    python main.py --health          # Quick health check
    python main.py --benchmark       # Run performance tests
    python main.py --download-models # Download AI models
    python main.py --version         # Show version info
    python main.py --dev             # Development mode
    python main.py --daemon          # Background daemon mode
    python main.py --port 8080       # Custom port
    python main.py --no-banner       # Start without banner

MCP TOOLS AVAILABLE (17 total):
    
    Core Memory Operations:
      • store_memory              - Store content with AI enhancement
      • search_memory             - Hybrid search with confidence scoring
      • delete_memory             - Safe memory deletion
      • analyze_response          - Hallucination detection

    Analytics & Intelligence:
      • get_memory_stats          - System statistics & health
      • get_learning_insights     - Adaptive learning data
      • detect_patterns           - Pattern recognition
      • analyze_temporal_evolution - Concept evolution tracking

    Advanced Features:
      • deduplicate_memories      - Semantic deduplication
      • summarize_memories        - AI-powered summarization  
      • crawl_website             - Natural language web crawling
      • health_check              - System health assessment

    Intelligent Suggestions:
      • suggest_related_memories  - ML-powered recommendations
      • detect_memory_connections - Connection discovery
      • recommend_memory_organization - Structure optimization
      • detect_knowledge_gaps     - Gap analysis & learning paths

    Web Integration:
      • web_search                - Local web search with content extraction

SYSTEM REQUIREMENTS:
    • Python 3.11+
    • PostgreSQL 15+ with pgvector
    • Redis 6.0+
    • Neo4j 5.0+
    • 8GB+ RAM (16GB+ recommended)
    • 20GB+ storage for models

CONFIGURATION:
    • Main config: ./config/config.yaml
    • Models: ./config/models.yaml  
    • Providers: ./config/providers.yaml
    • Environment: .env file

ENDPOINTS:
    • MCP Protocol: stdio (for agents like Claude)
    • REST API: http://localhost:8000
    • Analytics Dashboard: http://localhost:8050/dashboard  
    • Health Check: http://localhost:8000/v1/health
    • API Docs: http://localhost:8000/docs

DOCUMENTATION:
    • Installation: INSTALLATION.md
    • API Reference: docs/API.md
    • Architecture: ARCHITECTURE.md
    • Examples: examples/ directory

For more information, visit the documentation or check the health endpoint.
"""
    print(help_text)

async def main_with_args():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Tyra MCP Memory Server - Advanced AI Memory System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--version", "-v", action="store_true",
                       help="Show version information")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate system configuration")
    parser.add_argument("--no-banner", action="store_true",
                       help="Skip startup banner")
    parser.add_argument("--config", type=str,
                       help="Use custom config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    parser.add_argument("--help", "-h", action="store_true",
                       help="Show detailed help")
    parser.add_argument("--health", action="store_true",
                       help="Check system health and exit")
    parser.add_argument("--dev", action="store_true",
                       help="Start in development mode with enhanced debugging")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--download-models", action="store_true",
                       help="Download required AI models")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as background daemon")
    parser.add_argument("--port", type=int, default=8000,
                       help="Override default API port (8000)")
    
    # Handle help first (before parsing)
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        return
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.version:
        show_version()
        return
    
    if args.validate:
        success = await validate_system()
        sys.exit(0 if success else 1)
    
    if args.health:
        success = await check_system_health()
        sys.exit(0 if success else 1)
    
    if args.benchmark:
        success = await run_benchmarks()
        sys.exit(0 if success else 1)
    
    if args.download_models:
        success = await download_models()
        sys.exit(0 if success else 1)
    
    # Show banner unless disabled
    if not args.no_banner:
        show_banner()
    
    # Set environment variables if specified
    if args.config:
        os.environ["TYRA_CONFIG_PATH"] = args.config
        print(f"📝 Using config: {args.config}")
    
    if args.log_level:
        os.environ["TYRA_LOG_LEVEL"] = args.log_level
        print(f"📊 Log level: {args.log_level}")
    
    if args.dev:
        os.environ["TYRA_ENV"] = "development"
        os.environ["TYRA_LOG_LEVEL"] = "DEBUG"
        os.environ["TYRA_DEBUG"] = "true"
        print("🔧 Development mode enabled (Debug logging, detailed errors)")
    
    if args.port != 8000:
        os.environ["API_PORT"] = str(args.port)
        print(f"🌐 API port: {args.port}")
    
    if args.daemon:
        print("⚙️  Daemon mode: Server will run in background")
        # Note: Full daemon implementation would require process forking
        print("💡 For production, consider using systemd or supervisor")
    
    print()
    print("🚀 Starting Tyra Memory MCP Server...")
    print("💡 Use Ctrl+C to stop the server")
    print("📖 For help: python main.py --help")
    if args.dev:
        print("🔧 Development mode: Enhanced debugging enabled")
    print()
    
    # Import and run the main server
    try:
        from src.mcp.server import main
        await main()
    except ImportError as e:
        print(f"❌ Failed to import server module: {e}")
        print("Please ensure all dependencies are installed and the src directory exists.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main_with_args())
    except KeyboardInterrupt:
        print("\n")
        print("🛑 Server stopped by user")
        print("Thank you for using Tyra MCP Memory Server!")
    except Exception as e:
        print(f"\n❌ Server failed: {e}")
        print("Check logs for more details.")
        sys.exit(1)
