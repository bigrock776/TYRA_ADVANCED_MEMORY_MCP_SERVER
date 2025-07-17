# Tyra MCP Memory Server - Import Dependency Analysis
============================================================

## Summary Statistics
- Total Python files: 240
- Total imports: 2865
- Internal imports: 801
- External imports: 2064
- Unique external libraries: 127
- Orphaned modules: 114
- Circular dependencies: 11
- Files with syntax errors: 0

## Entry Points Analysis
### main.py
- Internal imports: 5
- External imports: 8
- Key imports:
  - src.mcp.server
  - src.mcp.server
  - src.mcp.server
  - src.core.memory.manager
  - src.core.memory.manager

### src/mcp/server.py
- Internal imports: 23
- External imports: 6
- Key imports:
  - mcp.server
  - mcp.server.models
  - mcp.server.stdio
  - mcp.types
  - core.adaptation.learning_engine
  - core.analytics.performance_tracker
  - core.memory.manager
  - core.rag.hallucination_detector
  - core.synthesis
  - core.clients.vllm_client

### src/api/app.py
- Internal imports: 33
- External imports: 10
- Key imports:
  - core.memory.manager
  - core.observability
  - core.utils.config
  - core.utils.logger
  - dashboard.main
  - middleware.auth
  - middleware.error_handler
  - middleware.rate_limit
  - websocket.server
  - websocket.memory_stream

## Most Imported Internal Modules
- src.core.utils.config: 74 imports
- src.core.cache.redis_cache: 56 imports
- src.core.utils.logger: 48 imports
- src.core.memory.manager: 37 imports
- src.core.utils.database: 36 imports
- src.models.memory: 26 imports
- src.core.embeddings.embedder: 25 imports
- src.core.utils.registry: 20 imports
- src.core.interfaces.graph_engine: 14 imports
- src.core.utils.simple_logger: 12 imports

## Orphaned Modules (Never Imported)
- analyze_imports
- main
- scripts.add_provider
- scripts.config_migrate
- scripts.migrate_config
- scripts.migrations.config
- scripts.migrations.config.001_initial_config
- scripts.migrations.config.002_add_observability
- scripts.migrations.config.003_add_self_learning
- scripts.migrations.config.004_add_trading_safety
- scripts.migrations.version_manager
- scripts.run_mcp_tests
- scripts.test_cross_encoder
- scripts.test_embedding_model
- scripts.test_model_pipeline
- scripts.test_setup
- scripts.validate_config
- src
- src.agents
- src.api
- src.api.middleware
- src.api.routes
- src.api.websocket
- src.clients
- src.core
- src.core.adaptation
- src.core.adaptation.self_training_scheduler
- src.core.agents.claude_integration
- src.core.ai.cross_component_validation
- src.core.analytics
- src.core.analytics.config_optimizer
- src.core.clients
- src.core.crawling
- src.core.embeddings
- src.core.embeddings.manager
- src.core.events
- src.core.events.trigger_system
- src.core.graph
- src.core.graph.enhanced_graph_client
- src.core.graph.recommender
- src.core.graph.temporal_evolution
- src.core.ingestion
- src.core.interfaces
- src.core.interfaces.hallucination_detector
- src.core.learning.ab_testing
- src.core.learning.continuous_improvement
- src.core.learning.hyperparameter_optimization
- src.core.memory
- src.core.memory.retriever
- src.core.memory.structured_operations
- src.core.prediction
- src.core.providers
- src.core.providers.embeddings
- src.core.providers.graph_engines
- src.core.providers.graph_engines.registry
- src.core.providers.rerankers
- src.core.providers.rerankers.registry
- src.core.providers.vector_stores
- src.core.providers.vector_stores.registry
- src.core.rag
- src.core.rag.retrieval
- src.core.rag.scorer
- src.core.services
- src.core.synthesis.deduplication
- src.core.synthesis.pattern_detector
- src.core.synthesis.summarization
- src.core.synthesis.temporal_analysis
- src.core.utils
- src.dashboard
- src.dashboard.analytics
- src.dashboard.insights
- src.dashboard.visualization
- src.ingest
- src.mcp
- src.mcp.tools
- src.mcp.transport
- src.memory
- src.migrations
- src.migrations.sql
- src.suggestions
- src.suggestions.connections
- src.suggestions.gaps
- src.suggestions.organization
- src.suggestions.related
- src.validators
- tests.integration.api.test_memory_endpoints
- tests.integration.test_end_to_end_workflows
- tests.integration.test_graph_integration
- tests.integration.test_provider_integration
- tests.performance
- tests.stress
- tests.test_basic
- tests.test_cache_manager
- tests.test_circuit_breaker
- tests.test_config_only
- tests.test_embeddings
- tests.test_graph_engine
- tests.test_hallucination_detector
- tests.test_ingestion
- tests.test_interface_validation
- tests.test_mcp_integration
- tests.test_mcp_server
- tests.test_mcp_simple
- tests.test_mcp_trading_safety
- tests.test_memory_manager
- tests.test_neo4j_engine
- tests.test_performance_tracker
- tests.test_phase2_config
- tests.test_reranking
- tests.test_server
- tests.test_vllm_reranker
- tests.unit.core.test_embeddings
- tests.unit.core.test_hallucination_detector
- tests.unit.core.test_memory_manager

## Circular Dependencies
### Cycle 1
src.validators.memory_confidence → src.memory.neo4j_linker → src.agents.websearch_agent → src.validators.memory_confidence

### Cycle 2
scripts.run_mcp_tests → src.mcp.server

### Cycle 3
tests.test_mcp_integration → src.mcp.server

### Cycle 4
src.api.app → src.api.routes.health → src.api.app

### Cycle 5
tests.test_mcp_trading_safety → src.mcp.server

### Cycle 6
src.api.routes.memory → src.api.app

### Cycle 7
src.api.routes.trading_data → src.api.app

### Cycle 8
src.api.routes.crawling → src.ingest.crawl4ai_runner

### Cycle 9
src.core.crawling.natural_language_parser → src.ingest.crawl4ai_runner

### Cycle 10
tests.integration.test_end_to_end_workflows → src.mcp.server

### Cycle 11
tests.integration.api.test_memory_endpoints → src.api.app

## External Libraries Used
 1. PIL
 2. abc
 3. aiohttp
 4. argparse
 5. ast
 6. asyncio
 7. asyncpg
 8. backoff
 9. base64
10. bcrypt
11. bs4
12. builtins
13. chardet
14. clip
15. collections
16. contextlib
17. contextvars
18. crawl4ai
19. cryptography
20. csv
21. cv2
22. dash
23. dash_bootstrap_components
24. dataclasses
25. datetime
26. dateutil
27. decimal
28. difflib
29. docx
30. duckduckgo_search
31. enum
32. fastapi
33. fitz
34. functools
35. gc
36. gensim
37. glob
38. graphiti_core
39. graphiti_neo4j
40. gzip
41. hashlib
42. heapq
43. hmac
44. html2text
45. httpx
46. implicit
47. importlib
48. inquirer
49. inspect
50. io
51. itertools
52. joblib
53. json
54. jwt
55. lightfm
56. logging
57. math
58. matplotlib
59. mimetypes
60. neo4j
61. networkx
62. newspaper
63. numpy
64. opentelemetry
65. operator
66. optuna
67. os
68. pandas
69. passlib
70. pathlib
71. peft
72. pgvector
73. pickle
74. plotly
75. psutil
76. pydantic
77. pydantic_ai
78. pydantic_settings
79. pygments
80. pytest
81. random
82. re
83. redis
84. rich
85. rouge_score
86. scipy
87. scripts
88. seaborn
89. secrets
90. sentence_transformers
91. shutil
92. signal
93. simple_config
94. simple_logger
95. sklearn
96. spacy
97. sqlalchemy
98. sqlparse
99. starlette
100. statistics
101. statsmodels
102. struct
103. structlog
104. subprocess
105. surprise
106. sys
107. tarfile
108. tempfile
109. threading
110. tiktoken
111. time
112. torch
113. traceback
114. trafilatura
115. transformers
116. types
117. typing
118. unittest
119. urllib
120. uuid
121. uvicorn
122. warnings
123. watchdog
124. weakref
125. websockets
126. yaml
127. zlib

## Import Trees from Entry Points
### main.py (main)
📁 main (2 imports)
  ├── src.mcp.server (20 imports)
    ├── src.core.utils.simple_logger (0 imports)
    ├── src.suggestions.gaps.local_detector (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── mcp.server.models (0 imports)
    ├── src.ingest.crawl4ai_runner (6 imports)
      ├── src.core.providers.embeddings.embedder (0 imports)
      ├── src.memory.pgvector_handler (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.validators.memory_confidence (0 imports)
      ├── src.memory.neo4j_linker (0 imports)
      ├── src.agents.websearch_agent (0 imports)
    ├── src.core.clients.vllm_client (2 imports)
      ├── src.core.utils.circuit_breaker (0 imports)
      ├── src.core.utils.config (0 imports)
    ├── src.core.memory.manager (7 imports)
      ├── src.core.interfaces.embeddings (0 imports)
      ├── src.core.interfaces.vector_store (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.interfaces.graph_engine (0 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.utils.registry (0 imports)
      ├── src.core.interfaces.reranker (0 imports)
    ├── src.core.crawling.natural_language_parser (2 imports)
      ├── src.core.utils.simple_logger (0 imports)
      ├── src.ingest.crawl4ai_runner (0 imports)
    ├── src.core.synthesis (4 imports)
      ├── src.core.deduplication (0 imports)
      ├── src.core.summarization (0 imports)
      ├── src.core.pattern_detector (0 imports)
      ├── src.core.temporal_analysis (0 imports)
    ├── mcp.types (0 imports)
    ├── src.suggestions.connections.local_connector (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── src.core.analytics.performance_tracker (0 imports)
    ├── src.core.adaptation.learning_engine (3 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.analytics.performance_tracker (0 imports)
      ├── src.core.utils.logger (0 imports)
    ├── src.core.utils.simple_config (0 imports)
    ├── mcp.server (0 imports)
    ├── src.core.rag.hallucination_detector (3 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.interfaces.embeddings (0 imports)
    ├── src.suggestions.related.local_suggester (3 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── src.suggestions.organization.local_recommender (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── mcp.server.stdio (0 imports)
    ├── src.core.services.file_watcher_service (2 imports)
      ├── src.core.utils.simple_logger (0 imports)
      ├── src.core.ingestion.file_watcher (0 imports)
    ├── src.agents.websearch_agent (6 imports)
      ├── src.core.providers.embeddings.embedder (0 imports)
      ├── src.memory.pgvector_handler (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.clients.vllm_client (0 imports)
      ├── src.validators.memory_confidence (0 imports)
      ├── src.memory.neo4j_linker (0 imports)
  ├── src.core.memory.manager (0 imports)

### src/mcp/server.py (src.mcp.server)
📁 src.mcp.server (20 imports)
  ├── src.core.utils.simple_logger (0 imports)
  ├── src.suggestions.gaps.local_detector (4 imports)
    ├── src.core.graph.neo4j_client (3 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.providers.graph_engines.neo4j (0 imports)
      ├── src.core.interfaces.graph_engine (0 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.embeddings.embedder (0 imports)
    ├── src.core.memory.manager (7 imports)
      ├── src.core.interfaces.embeddings (0 imports)
      ├── src.core.interfaces.vector_store (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.interfaces.graph_engine (0 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.utils.registry (0 imports)
      ├── src.core.interfaces.reranker (0 imports)
  ├── mcp.server.models (0 imports)
  ├── src.ingest.crawl4ai_runner (6 imports)
    ├── src.core.providers.embeddings.embedder (5 imports)
      ├── src.core.providers.embeddings.registry (0 imports)
      ├── src.core.interfaces.embeddings (0 imports)
      ├── src.utils.config (0 imports)
      ├── src.core.observability.tracing (0 imports)
      ├── src.utils.logger (0 imports)
    ├── src.memory.pgvector_handler (1 imports)
      ├── src.core.utils.config (0 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.validators.memory_confidence (2 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.memory.neo4j_linker (0 imports)
    ├── src.memory.neo4j_linker (2 imports)
      ├── src.agents.websearch_agent (0 imports)
      ├── src.core.utils.config (0 imports)
    ├── src.agents.websearch_agent (6 imports)
      ├── src.core.providers.embeddings.embedder (0 imports)
      ├── src.memory.pgvector_handler (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.clients.vllm_client (0 imports)
      ├── src.validators.memory_confidence (0 imports)
      ├── src.memory.neo4j_linker (0 imports)
  ├── src.core.clients.vllm_client (2 imports)
    ├── src.core.utils.circuit_breaker (1 imports)
      ├── src.core.utils.logger (0 imports)
    ├── src.core.utils.config (0 imports)
  ├── src.core.memory.manager (0 imports)
  ├── src.core.crawling.natural_language_parser (2 imports)
    ├── src.core.utils.simple_logger (0 imports)
    ├── src.ingest.crawl4ai_runner (0 imports)
  ├── src.core.synthesis (4 imports)
    ├── src.core.deduplication (0 imports)
    ├── src.core.summarization (0 imports)
    ├── src.core.pattern_detector (0 imports)
    ├── src.core.temporal_analysis (0 imports)
  ├── mcp.types (0 imports)
  ├── src.suggestions.connections.local_connector (4 imports)
    ├── src.core.graph.neo4j_client (0 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.embeddings.embedder (0 imports)
    ├── src.core.memory.manager (0 imports)
  ├── src.core.analytics.performance_tracker (0 imports)
  ├── src.core.adaptation.learning_engine (3 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.analytics.performance_tracker (0 imports)
    ├── src.core.utils.logger (1 imports)
      ├── src.core.utils.config (0 imports)
  ├── src.core.utils.simple_config (0 imports)
  ├── mcp.server (0 imports)
  ├── src.core.rag.hallucination_detector (3 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.interfaces.embeddings (0 imports)
  ├── src.suggestions.related.local_suggester (3 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.embeddings.embedder (0 imports)
    ├── src.core.memory.manager (0 imports)
  ├── src.suggestions.organization.local_recommender (4 imports)
    ├── src.core.graph.neo4j_client (0 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.embeddings.embedder (0 imports)
    ├── src.core.memory.manager (0 imports)
  ├── mcp.server.stdio (0 imports)
  ├── src.core.services.file_watcher_service (2 imports)
    ├── src.core.utils.simple_logger (0 imports)
    ├── src.core.ingestion.file_watcher (4 imports)
      ├── src.core.ingestion.file_loaders (0 imports)
      ├── src.core.utils.simple_logger (0 imports)
      ├── src.core.utils.simple_config (0 imports)
      ├── src.core.ingestion.document_processor (0 imports)
  ├── src.agents.websearch_agent (0 imports)

### src/api/app.py (src.api.app)
📁 src.api.app (33 imports)
  ├── src.api.routes.search (4 imports)
    ├── src.core.utils.logger (1 imports)
      ├── src.core.utils.config (0 imports)
    ├── src.core.search.searcher (0 imports)
    ├── src.core.memory.models (0 imports)
    ├── src.core.utils.registry (2 imports)
      ├── src.core.utils.simple_logger (0 imports)
      ├── src.core.utils.config (0 imports)
  ├── src.api.middleware.auth (2 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.utils.logger (0 imports)
  ├── src.api.routes.suggestions (7 imports)
    ├── src.suggestions.related.local_suggester (3 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── src.core.providers.embeddings.embedder (5 imports)
      ├── src.core.providers.embeddings.registry (0 imports)
      ├── src.core.interfaces.embeddings (0 imports)
      ├── src.utils.config (0 imports)
      ├── src.core.observability.tracing (0 imports)
      ├── src.utils.logger (0 imports)
    ├── src.suggestions.gaps.local_detector (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── src.suggestions.organization.local_recommender (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── src.suggestions.connections.local_connector (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.memory.manager (7 imports)
      ├── src.core.interfaces.embeddings (0 imports)
      ├── src.core.interfaces.vector_store (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.interfaces.graph_engine (0 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.utils.registry (0 imports)
      ├── src.core.interfaces.reranker (0 imports)
  ├── src.api.routes.analytics (6 imports)
    ├── src.core.adaptation.memory_health (0 imports)
    ├── src.core.utils.rate_limiter (0 imports)
    ├── src.core.adaptation.config_optimizer (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.analytics.performance_tracker (0 imports)
    ├── src.core.utils.auth (0 imports)
  ├── src.api.routes.chat (6 imports)
    ├── src.core.rag.hallucination_detector (3 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.interfaces.embeddings (0 imports)
    ├── src.core.search.searcher (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.rag.reranker (6 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.utils.circuit_breaker (0 imports)
      ├── src.core.rag.vllm_reranker (0 imports)
      ├── src.core.observability.tracing (0 imports)
      ├── src.core.interfaces.reranker (0 imports)
      ├── src.utils.logger (0 imports)
    ├── src.core.memory.manager (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.api.routes.file_watcher (5 imports)
    ├── src.core.utils.simple_logger (0 imports)
    ├── src.api.middleware.rate_limit (4 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.utils.registry (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.api.middleware.auth (0 imports)
    ├── src.core.services.file_watcher_service (2 imports)
      ├── src.core.utils.simple_logger (0 imports)
      ├── src.core.ingestion.file_watcher (0 imports)
    ├── src.core.utils.simple_config (0 imports)
  ├── src.api.routes.synthesis (7 imports)
    ├── src.core.synthesis (4 imports)
      ├── src.core.deduplication (0 imports)
      ├── src.core.summarization (0 imports)
      ├── src.core.pattern_detector (0 imports)
      ├── src.core.temporal_analysis (0 imports)
    ├── src.core.rag.hallucination_detector (0 imports)
    ├── src.core.providers.embeddings.embedder (0 imports)
    ├── src.core.clients.vllm_client (2 imports)
      ├── src.core.utils.circuit_breaker (0 imports)
      ├── src.core.utils.config (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.memory.manager (0 imports)
    ├── src.models.memory (0 imports)
  ├── src.api.routes.admin (8 imports)
    ├── src.core.observability.telemetry (4 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.observability.performance_optimized_telemetry (0 imports)
      ├── src.core.observability.telemetry_optimizer (0 imports)
    ├── src.core.cache.redis_cache (3 imports)
      ├── src.core.utils.database (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.utils.logger (0 imports)
    ├── src.core.schemas.ingestion (0 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.ingestion.job_tracker (2 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.memory.manager (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.core.memory.manager (0 imports)
  ├── src.api.websocket.memory_stream (3 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.api.websocket.server (2 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.models.memory (0 imports)
  ├── src.api.routes.health (2 imports)
    ├── src.api.app (0 imports)
    ├── src.core.utils.logger (0 imports)
  ├── src.api.routes.personalization (5 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.personalization.confidence.adaptive_confidence (2 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.personalization.preferences.user_preference_engine (2 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.core.utils.config (0 imports)
  ├── src.api.routes.observability (7 imports)
    ├── src.core.observability.telemetry (0 imports)
    ├── src.core.observability.dashboard (3 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.observability.tracing (2 imports)
      ├── src.core.observability.telemetry (0 imports)
      ├── src.core.utils.logger (0 imports)
    ├── src.core.observability.metrics (2 imports)
      ├── src.core.observability.telemetry (0 imports)
      ├── src.core.utils.logger (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.dashboard.main (6 imports)
    ├── src.core.utils.simple_logger (0 imports)
    ├── src.dashboard.insights.local_gaps (5 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.suggestions.gaps.local_detector (0 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── src.dashboard.visualization.local_network (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.memory.manager (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.dashboard.analytics.local_usage (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.memory.manager (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.dashboard.analytics.local_roi (4 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.memory.manager (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.memory.manager (0 imports)
  ├── src.api.routes.security (6 imports)
    ├── src.core.security.audit (2 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.security.encryption (3 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.security.rbac (2 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.api.websocket.server (0 imports)
  ├── src.api.routes.ingestion (8 imports)
    ├── src.core.schemas.ingestion (0 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.ingestion.job_tracker (0 imports)
    ├── src.core.ingestion.document_processor (7 imports)
      ├── src.core.schemas.ingestion (0 imports)
      ├── src.core.ingestion.chunking_strategies (0 imports)
      ├── src.core.ingestion.llm_context_enhancer (0 imports)
      ├── src.core.observability (0 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.ingestion.file_loaders (0 imports)
      ├── src.core.memory.manager (0 imports)
    ├── src.core.observability (3 imports)
      ├── src.core.metrics (0 imports)
      ├── src.core.tracing (0 imports)
      ├── src.core.telemetry (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.ingestion.file_loaders (1 imports)
      ├── src.core.utils.logger (0 imports)
    ├── src.core.memory.manager (0 imports)
  ├── src.api.websocket.search_stream (4 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.embeddings.embedder (0 imports)
    ├── src.api.websocket.server (0 imports)
    ├── src.models.memory (0 imports)
  ├── src.api.routes.embeddings (8 imports)
    ├── src.core.embeddings.multi_perspective (3 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.embeddings.fusion (5 imports)
      ├── src.core.embeddings.multi_perspective (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.embeddings.contextual (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.embeddings.embedder (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.embeddings.contextual (3 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.embeddings.fine_tuning (3 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.api.middleware.error_handler (2 imports)
    ├── src.core.utils.config (0 imports)
    ├── src.core.utils.logger (0 imports)
  ├── src.api.routes.trading_data (3 imports)
    ├── src.api.app (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.memory.manager (0 imports)
  ├── src.api.routes.crawling (6 imports)
    ├── src.core.utils.simple_logger (0 imports)
    ├── src.api.middleware.rate_limit (0 imports)
    ├── src.api.middleware.auth (0 imports)
    ├── src.ingest.crawl4ai_runner (6 imports)
      ├── src.core.providers.embeddings.embedder (0 imports)
      ├── src.memory.pgvector_handler (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.validators.memory_confidence (0 imports)
      ├── src.memory.neo4j_linker (0 imports)
      ├── src.agents.websearch_agent (0 imports)
    ├── src.core.memory.manager (0 imports)
    ├── src.core.crawling.natural_language_parser (2 imports)
      ├── src.core.utils.simple_logger (0 imports)
      ├── src.ingest.crawl4ai_runner (0 imports)
  ├── src.api.routes.webhooks (4 imports)
    ├── src.core.agents.agent_logger (1 imports)
      ├── src.core.utils.logger (0 imports)
    ├── src.api.dependencies (0 imports)
    ├── src.core.agents.session_manager (3 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.memory.manager (0 imports)
  ├── src.api.middleware.rate_limit (0 imports)
  ├── src.api.routes.prediction (8 imports)
    ├── src.core.prediction.usage_analyzer (3 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.prediction.lifecycle (5 imports)
      ├── src.core.prediction.usage_analyzer (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.prediction.auto_archiver (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.prediction.auto_archiver (4 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.prediction.usage_analyzer (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.prediction.preloader (4 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.prediction.usage_analyzer (0 imports)
      ├── src.models.memory (0 imports)
    ├── src.core.memory.manager (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.core.services.file_watcher_service (0 imports)
  ├── src.api.routes.memory (3 imports)
    ├── src.api.app (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.memory.manager (0 imports)
  ├── src.core.observability (0 imports)
  ├── src.core.utils.logger (0 imports)
  ├── src.api.routes.rag (10 imports)
    ├── src.core.rag.hallucination_detector (0 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.graph.neo4j_client (3 imports)
      ├── src.core.utils.logger (0 imports)
      ├── src.core.providers.graph_engines.neo4j (0 imports)
      ├── src.core.interfaces.graph_engine (0 imports)
    ├── src.core.embeddings.embedder (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.rag.pipeline_manager (9 imports)
      ├── src.core.rag.hallucination_detector (0 imports)
      ├── src.core.rag.intent_detector (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.graph.neo4j_client (0 imports)
      ├── src.core.rag.chunk_linking (0 imports)
      ├── src.core.embeddings.embedder (0 imports)
      ├── src.core.rag.multimodal (0 imports)
      ├── src.core.memory.manager (0 imports)
      ├── src.core.rag.dynamic_reranking (0 imports)
    ├── core.graph.neo4j_client (0 imports)
    ├── src.core.rag.reranker (0 imports)
    ├── src.core.memory.manager (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.api.routes.graph (5 imports)
    ├── src.core.graph.graphiti_integration (4 imports)
      ├── src.utils.circuit_breaker (0 imports)
      ├── core.memory.manager (0 imports)
      ├── src.interfaces.graph_engine (0 imports)
      ├── src.utils.logger (0 imports)
    ├── src.core.graph.neo4j_client (0 imports)
    ├── src.core.graph.causal_inference (4 imports)
      ├── src.core.cache.redis_cache (0 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.models.memory (0 imports)
      ├── src.core.graph.reasoning_engine (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.utils.registry (0 imports)
  ├── src.api.routes.performance (5 imports)
    ├── src.core.cache.redis_cache (0 imports)
    ├── src.infrastructure.scaling.auto_scaler (2 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.utils.logger (0 imports)
    ├── src.core.optimization.query_optimizer (2 imports)
      ├── src.core.utils.config (0 imports)
      ├── src.core.cache.redis_cache (0 imports)
    ├── src.core.utils.registry (0 imports)

## Detailed File Analysis
### analyze_imports.py
- Module: analyze_imports
- Entry point: False
- Functions: 16
- Classes: 3
- Total imports: 9
- Internal imports: 0
- External imports: 9
- External imports:
  - import ast
  - import os
  - import sys
  - from collections import defaultdict, deque
  - from dataclasses import dataclass, field
  - ... and 4 more

### main.py
- Module: main
- Entry point: True
- Functions: 3
- Classes: 0
- Total imports: 13
- Internal imports: 5
- External imports: 8
- Internal imports:
  - from src.mcp.server import TyraMemoryServer
  - from src.mcp.server import TyraMemoryServer
  - from src.mcp.server import main
  - from src.core.memory.manager import MemoryStoreRequest
  - from src.core.memory.manager import MemorySearchRequest
- External imports:
  - import argparse
  - import asyncio
  - import os
  - import sys
  - from datetime import datetime
  - ... and 3 more

### scripts/add_provider.py
- Module: scripts.add_provider
- Entry point: False
- Functions: 16
- Classes: 1
- Total imports: 10
- Internal imports: 2
- External imports: 8
- Internal imports:
  - from src.core.utils.registry import ProviderType
  - from src.core.utils.registry import provider_registry
- External imports:
  - import argparse
  - import os
  - import sys
  - import yaml
  - from pathlib import Path
  - ... and 3 more

### scripts/config_migrate.py
- Module: scripts.config_migrate
- Entry point: False
- Functions: 12
- Classes: 1
- Total imports: 16
- Internal imports: 1
- External imports: 15
- Internal imports:
  - from src.core.utils.simple_logger import get_logger
- External imports:
  - import argparse
  - import json
  - import os
  - import shutil
  - import sys
  - ... and 10 more

### scripts/migrate_config.py
- Module: scripts.migrate_config
- Entry point: False
- Functions: 21
- Classes: 1
- Total imports: 11
- Internal imports: 2
- External imports: 9
- Internal imports:
  - from src.core.utils.simple_config import SimpleConfig
  - from src.core.utils.simple_logger import get_logger
- External imports:
  - import argparse
  - import json
  - import os
  - import shutil
  - import sys
  - ... and 4 more

### scripts/run_mcp_tests.py
- Module: scripts.run_mcp_tests
- Entry point: False
- Functions: 8
- Classes: 1
- Total imports: 11
- Internal imports: 1
- External imports: 10
- Internal imports:
  - from src.mcp.server import TyraMemoryServer
- External imports:
  - import asyncio
  - import argparse
  - import json
  - import sys
  - import time
  - ... and 5 more

### scripts/test_cross_encoder.py
- Module: scripts.test_cross_encoder
- Entry point: False
- Functions: 3
- Classes: 0
- Total imports: 8
- Internal imports: 1
- External imports: 7
- Internal imports:
  - from core.utils.logger import get_logger
- External imports:
  - import os
  - import sys
  - import time
  - from pathlib import Path
  - import numpy
  - ... and 2 more

### scripts/test_embedding_model.py
- Module: scripts.test_embedding_model
- Entry point: False
- Functions: 3
- Classes: 0
- Total imports: 8
- Internal imports: 1
- External imports: 7
- Internal imports:
  - from core.utils.logger import get_logger
- External imports:
  - import os
  - import sys
  - import time
  - from pathlib import Path
  - import numpy
  - ... and 2 more

### scripts/test_model_pipeline.py
- Module: scripts.test_model_pipeline
- Entry point: False
- Functions: 3
- Classes: 0
- Total imports: 8
- Internal imports: 1
- External imports: 7
- Internal imports:
  - from core.utils.logger import get_logger
- External imports:
  - import os
  - import sys
  - import time
  - from pathlib import Path
  - import numpy
  - ... and 2 more

### scripts/test_setup.py
- Module: scripts.test_setup
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 10
- Internal imports: 3
- External imports: 7
- Internal imports:
  - from core.utils.simple_config import get_setting
  - from core.utils.simple_config import get_setting
  - from core.utils.simple_config import get_setting
- External imports:
  - import asyncio
  - import os
  - import sys
  - from pathlib import Path
  - import asyncpg
  - ... and 2 more

### scripts/validate_config.py
- Module: scripts.validate_config
- Entry point: False
- Functions: 14
- Classes: 1
- Total imports: 7
- Internal imports: 1
- External imports: 6
- Internal imports:
  - from src.core.utils.simple_logger import get_logger
- External imports:
  - import argparse
  - import json
  - import sys
  - import yaml
  - from pathlib import Path
  - ... and 1 more

### src/__init__.py
- Module: src
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 5
- Internal imports: 5
- External imports: 0
- Internal imports:
  - from core.utils.simple_logger import get_logger
  - from core.interfaces.embeddings import EmbeddingProvider
  - from core.interfaces.graph_engine import GraphEngine
  - from core.interfaces.vector_store import VectorStore
  - from core.memory.manager import MemoryManager

### tests/test_basic.py
- Module: tests.test_basic
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 9
- Internal imports: 3
- External imports: 6
- Internal imports:
  - from src.core.utils.simple_config import get_setting, get_settings
  - from src.core.utils.simple_logger import get_logger
  - from src.core.utils.simple_config import get_setting
- External imports:
  - import asyncio
  - import json
  - import os
  - import sys
  - from pathlib import Path
  - ... and 1 more

### tests/test_cache_manager.py
- Module: tests.test_cache_manager
- Entry point: False
- Functions: 0
- Classes: 1
- Total imports: 5
- Internal imports: 1
- External imports: 4
- Internal imports:
  - from src.core.cache.manager import CacheManager, CacheKey, CacheEntry
- External imports:
  - import pytest
  - import asyncio
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import List, Dict, Any, Optional

### tests/test_circuit_breaker.py
- Module: tests.test_circuit_breaker
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 5
- Internal imports: 1
- External imports: 4
- Internal imports:
  - from src.core.utils.circuit_breaker import CircuitBreaker, CircuitBreakerState, CircuitBreakerConfig
- External imports:
  - import pytest
  - import asyncio
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import List, Dict, Any, Optional

### tests/test_config_only.py
- Module: tests.test_config_only
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 12
- Internal imports: 2
- External imports: 10
- Internal imports:
  - from mcp.server import Server
  - from mcp.types import CallToolResult, TextContent, Tool
- External imports:
  - import os
  - import sys
  - from pathlib import Path
  - import asyncio
  - from simple_config import get_setting, get_settings
  - ... and 5 more

### tests/test_embeddings.py
- Module: tests.test_embeddings
- Entry point: False
- Functions: 0
- Classes: 1
- Total imports: 6
- Internal imports: 1
- External imports: 5
- Internal imports:
  - from src.core.embeddings.embedder import EmbeddingProvider, EmbeddingRequest, EmbeddingResult
- External imports:
  - import pytest
  - import numpy
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import List
  - import asyncio

### tests/test_graph_engine.py
- Module: tests.test_graph_engine
- Entry point: False
- Functions: 0
- Classes: 1
- Total imports: 4
- Internal imports: 1
- External imports: 3
- Internal imports:
  - from src.core.graph.engine import GraphEngine, EntityExtractionRequest, GraphQueryRequest
- External imports:
  - import pytest
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import List, Dict, Any

### tests/test_hallucination_detector.py
- Module: tests.test_hallucination_detector
- Entry point: False
- Functions: 0
- Classes: 1
- Total imports: 4
- Internal imports: 1
- External imports: 3
- Internal imports:
  - from src.core.rag.hallucination_detector import HallucinationDetector, HallucinationResult
- External imports:
  - import pytest
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import List, Dict, Any

### tests/test_ingestion.py
- Module: tests.test_ingestion
- Entry point: False
- Functions: 15
- Classes: 8
- Total imports: 15
- Internal imports: 6
- External imports: 9
- Internal imports:
  - from src.api.app import create_app
  - from src.core.ingestion.chunking_strategies import AutoChunkingStrategy, ParagraphChunkingStrategy, SemanticChunkingStrategy, chunk_content
  - from src.core.ingestion.document_processor import DocumentProcessor
  - from src.core.ingestion.file_loaders import CSVLoader, DOCXLoader, HTMLLoader, JSONLoader, PDFLoader, TextLoader, get_file_loader
  - from src.core.ingestion.llm_context_enhancer import LLMContextEnhancer
  - from src.core.schemas.ingestion import IngestRequest, IngestResponse
- External imports:
  - import asyncio
  - import base64
  - import io
  - import json
  - import tempfile
  - ... and 4 more

### tests/test_interface_validation.py
- Module: tests.test_interface_validation
- Entry point: False
- Functions: 11
- Classes: 8
- Total imports: 23
- Internal imports: 15
- External imports: 8
- Internal imports:
  - from src.core.interfaces.embeddings import EmbeddingProvider, EmbeddingProviderError, EmbeddingInitializationError, EmbeddingGenerationError, EmbeddingConfigurationError
  - from src.core.interfaces.vector_store import VectorStore, VectorDocument, VectorSearchResult, VectorStoreError, VectorStoreInitializationError, VectorStoreOperationError
  - from src.core.interfaces.graph_engine import GraphEngine, Entity, Relationship, RelationshipType, GraphEngineError, GraphEngineInitializationError, GraphEngineOperationError
  - from src.core.interfaces.reranker import Reranker, RerankingCandidate, RerankingResult, RerankerType, RerankerError, RerankerInitializationError, RerankerOperationError
  - from src.core.providers.embeddings.huggingface import HuggingFaceProvider
  - from src.core.providers.vector_stores.pgvector import PgVectorStore
  - from src.core.providers.graph_engines.neo4j import Neo4jEngine
  - from src.core.providers.rerankers.cross_encoder import CrossEncoderReranker
  - from src.core.providers.graph_engines.neo4j import Neo4jEngine
  - from src.core.providers.graph_engines.neo4j import Neo4jEngine
  - from src.core.providers.graph_engines.neo4j import Neo4jEngine
  - from src.core.providers.rerankers.cross_encoder import CrossEncoderReranker
  - from src.core.providers.rerankers.cross_encoder import CrossEncoderReranker
  - from src.core.providers.rerankers.cross_encoder import CrossEncoderReranker
  - from src.core.providers.embeddings.nonexistent import NonexistentProvider
- External imports:
  - import asyncio
  - import inspect
  - import pytest
  - from abc import ABC, abstractmethod
  - from typing import Any, Dict, List, Optional, Type, Union
  - ... and 3 more

### tests/test_mcp_integration.py
- Module: tests.test_mcp_integration
- Entry point: False
- Functions: 3
- Classes: 2
- Total imports: 9
- Internal imports: 2
- External imports: 7
- Internal imports:
  - from mcp.types import CallToolRequest, CallToolResult, TextContent
  - from src.mcp.server import TyraMemoryServer
- External imports:
  - import asyncio
  - import json
  - import pytest
  - import uuid
  - from datetime import datetime
  - ... and 2 more

### tests/test_mcp_server.py
- Module: tests.test_mcp_server
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 12
- Internal imports: 8
- External imports: 4
- Internal imports:
  - from mcp.server import Server
  - from mcp.server.models import InitializationOptions
  - from mcp.server.stdio import stdio_server
  - from mcp.types import CallToolRequest, CallToolResult, TextContent, Tool
  - from mcp.server import Server
  - from mcp.types import Tool
  - from mcp.server import Server
  - from mcp.types import CallToolResult, TextContent, Tool
- External imports:
  - import asyncio
  - import sys
  - from pathlib import Path
  - import traceback

### tests/test_mcp_simple.py
- Module: tests.test_mcp_simple
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 3
- Internal imports: 2
- External imports: 1
- Internal imports:
  - from mcp.server import Server
  - from mcp.types import Tool
- External imports:
  - import asyncio

### tests/test_mcp_trading_safety.py
- Module: tests.test_mcp_trading_safety
- Entry point: False
- Functions: 0
- Classes: 1
- Total imports: 4
- Internal imports: 1
- External imports: 3
- Internal imports:
  - from src.mcp.server import TyraMemoryServer
- External imports:
  - import json
  - import pytest
  - from unittest.mock import AsyncMock, MagicMock

### tests/test_memory_manager.py
- Module: tests.test_memory_manager
- Entry point: False
- Functions: 0
- Classes: 1
- Total imports: 7
- Internal imports: 2
- External imports: 5
- Internal imports:
  - from src.core.memory.manager import MemoryManager, MemoryStoreRequest, MemorySearchRequest
  - from src.core.memory.models import Memory, MemoryMetadata
- External imports:
  - import pytest
  - import uuid
  - from datetime import datetime
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import List, Dict, Any

### tests/test_neo4j_engine.py
- Module: tests.test_neo4j_engine
- Entry point: False
- Functions: 2
- Classes: 10
- Total imports: 10
- Internal imports: 2
- External imports: 8
- Internal imports:
  - from src.core.providers.graph_engines.neo4j import Neo4jEngine
  - from src.core.interfaces.graph_engine import Entity, Relationship, GraphEngineError, GraphEngineInitializationError, GraphEngineOperationError
- External imports:
  - import asyncio
  - import json
  - import pytest
  - import uuid
  - from datetime import datetime, timedelta
  - ... and 3 more

### tests/test_performance_tracker.py
- Module: tests.test_performance_tracker
- Entry point: False
- Functions: 0
- Classes: 1
- Total imports: 6
- Internal imports: 1
- External imports: 5
- Internal imports:
  - from src.core.analytics.performance_tracker import PerformanceTracker, MetricType, PerformanceMetric
- External imports:
  - import pytest
  - import time
  - from datetime import datetime, timedelta
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import List, Dict, Any, Optional

### tests/test_phase2_config.py
- Module: tests.test_phase2_config
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 13
- Internal imports: 7
- External imports: 6
- Internal imports:
  - from src.core.utils.config import ConfigManager, get_settings, reload_settings
  - from src.core.utils.database import Neo4jManager, PostgreSQLManager, RedisManager
  - from src.core.utils.logger import get_logger
  - from src.core.utils.logger import clear_request_context, set_request_context
  - from src.core.utils.logger import log_performance
  - from src.core.utils.circuit_breaker import AsyncCircuitBreaker, CircuitBreakerConfig
  - from src.core.utils.registry import ProviderType, provider_registry
- External imports:
  - import asyncio
  - import os
  - import sys
  - from pathlib import Path
  - import traceback
  - ... and 1 more

### tests/test_reranking.py
- Module: tests.test_reranking
- Entry point: False
- Functions: 0
- Classes: 1
- Total imports: 4
- Internal imports: 1
- External imports: 3
- Internal imports:
  - from src.core.rag.reranker import Reranker, RerankRequest, RerankResult
- External imports:
  - import pytest
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import List, Dict, Any

### tests/test_server.py
- Module: tests.test_server
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 7
- Internal imports: 2
- External imports: 5
- Internal imports:
  - from src.core.utils.simple_config import get_setting, get_settings
  - from src.mcp_server.server import TyraMemoryServer
- External imports:
  - import asyncio
  - import json
  - import os
  - import sys
  - from pathlib import Path

### tests/test_vllm_reranker.py
- Module: tests.test_vllm_reranker
- Entry point: False
- Functions: 4
- Classes: 1
- Total imports: 6
- Internal imports: 1
- External imports: 5
- Internal imports:
  - from src.core.rag.vllm_reranker import VLLMReranker
- External imports:
  - import pytest
  - from unittest.mock import AsyncMock, MagicMock, patch
  - import aiohttp
  - import asyncio
  - from typing import Dict, Any

### scripts/migrations/version_manager.py
- Module: scripts.migrations.version_manager
- Entry point: False
- Functions: 10
- Classes: 1
- Total imports: 4
- Internal imports: 0
- External imports: 4
- External imports:
  - import json
  - from datetime import datetime
  - from pathlib import Path
  - from typing import Dict, List, Optional, Any

### src/agents/__init__.py
- Module: src.agents
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from websearch_agent import WebSearchAgent, WebSearchResult, SearchMethod, ContentExtractor, LocalWebSearcher

### src/agents/websearch_agent.py
- Module: src.agents.websearch_agent
- Entry point: False
- Functions: 8
- Classes: 7
- Total imports: 21
- Internal imports: 6
- External imports: 15
- Internal imports:
  - from core.clients.vllm_client import VLLMClient, ChatMessage, Role
  - from core.providers.embeddings.embedder import Embedder
  - from memory.pgvector_handler import PgVectorHandler
  - from memory.neo4j_linker import Neo4jLinker
  - from validators.memory_confidence import MemoryConfidenceAgent
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import hashlib
  - import re
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Set, Any, Union, Tuple
  - ... and 10 more

### src/api/__init__.py
- Module: src.api
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from app import create_app, get_app

### src/api/app.py
- Module: src.api.app
- Entry point: True
- Functions: 6
- Classes: 1
- Total imports: 43
- Internal imports: 33
- External imports: 10
- Internal imports:
  - from core.memory.manager import MemoryManager
  - from core.observability import get_memory_metrics, get_telemetry, get_tracer
  - from core.utils.config import get_settings
  - from core.utils.logger import get_logger
  - from dashboard.main import create_dashboard_app
  - from middleware.auth import AuthMiddleware
  - from middleware.error_handler import ErrorHandlerMiddleware
  - from middleware.rate_limit import RateLimitMiddleware
  - from websocket.server import WebSocketManager
  - from websocket.memory_stream import MemoryStreamHandler
  - from websocket.search_stream import SearchStreamHandler
  - from core.services.file_watcher_service import get_file_watcher_manager
  - from routes.admin import router
  - from routes.analytics import router
  - from routes.chat import router
  - from routes.crawling import router
  - from routes.embeddings import router
  - from routes.file_watcher import router
  - from routes.graph import router
  - from routes.health import router
  - from routes.ingestion import router
  - from routes.memory import router
  - from routes.observability import router
  - from routes.performance import router
  - from routes.personalization import router
  - from routes.prediction import router
  - from routes.rag import router
  - from routes.search import router
  - from routes.security import router
  - from routes.suggestions import router
  - from routes.synthesis import router
  - from routes.trading_data import router
  - from routes.webhooks import router
- External imports:
  - import logging
  - import time
  - import uuid
  - from contextlib import asynccontextmanager
  - from typing import Any, Dict, Optional
  - ... and 5 more

### src/clients/__init__.py
- Module: src.clients
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/clients/memory_client.py
- Module: src.clients.memory_client
- Entry point: False
- Functions: 5
- Classes: 6
- Total imports: 12
- Internal imports: 4
- External imports: 8
- Internal imports:
  - from core.cache import CacheLevel, RedisCache
  - from core.observability import get_memory_metrics, get_tracer
  - from core.utils.config import get_settings
  - from core.utils.logger import get_logger
- External imports:
  - import asyncio
  - import json
  - import time
  - from dataclasses import asdict, dataclass
  - from datetime import datetime
  - ... and 3 more

### src/core/__init__.py
- Module: src.core
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/dashboard/__init__.py
- Module: src.dashboard
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/dashboard/main.py
- Module: src.dashboard.main
- Entry point: False
- Functions: 1
- Classes: 0
- Total imports: 12
- Internal imports: 6
- External imports: 6
- Internal imports:
  - from analytics.local_usage import UsageAnalyticsDashboard
  - from analytics.local_roi import ROIAnalyticsDashboard
  - from insights.local_gaps import GapAnalyticsDashboard
  - from visualization.local_network import NetworkVisualizationDashboard
  - from core.memory.manager import MemoryManager
  - from core.utils.simple_logger import get_logger
- External imports:
  - import asyncio
  - from typing import Dict, Any, Optional
  - from fastapi import FastAPI, Request, Depends
  - from fastapi.responses import HTMLResponse, JSONResponse
  - from fastapi.staticfiles import StaticFiles
  - ... and 1 more

### src/ingest/__init__.py
- Module: src.ingest
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from crawl4ai_runner import Crawl4aiRunner, CrawlStrategy, ContentType, CrawlStatus, CrawlResult, DomainPolicy

### src/ingest/crawl4ai_runner.py
- Module: src.ingest.crawl4ai_runner
- Entry point: False
- Functions: 5
- Classes: 6
- Total imports: 21
- Internal imports: 6
- External imports: 15
- Internal imports:
  - from core.providers.embeddings.embedder import Embedder
  - from memory.pgvector_handler import PgVectorHandler
  - from memory.neo4j_linker import Neo4jLinker
  - from agents.websearch_agent import WebSearchResult, ContentExtractor, ExtractionQuality
  - from validators.memory_confidence import MemoryConfidenceAgent
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import hashlib
  - import json
  - import re
  - from datetime import datetime, timedelta
  - ... and 10 more

### src/mcp/__init__.py
- Module: src.mcp
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from server import TyraMemoryServer, main

### src/mcp/server.py
- Module: src.mcp.server
- Entry point: True
- Functions: 2
- Classes: 1
- Total imports: 29
- Internal imports: 23
- External imports: 6
- Internal imports:
  - from mcp.server import Server
  - from mcp.server.models import InitializationOptions
  - from mcp.server.stdio import stdio_server
  - from mcp.types import CallToolRequest, CallToolResult, GetPromptRequest, GetPromptResult, ListToolsRequest, Prompt, PromptMessage, Role, TextContent, Tool
  - from core.adaptation.learning_engine import LearningEngine
  - from core.analytics.performance_tracker import MetricType, PerformanceTracker
  - from core.memory.manager import MemoryManager, MemorySearchRequest, MemoryStoreRequest
  - from core.rag.hallucination_detector import HallucinationDetector
  - from core.synthesis import DeduplicationEngine, SummarizationEngine, PatternDetector, TemporalAnalyzer, SummarizationType
  - from core.clients.vllm_client import VLLMClient
  - from core.utils.simple_config import get_setting, get_settings
  - from core.utils.simple_logger import get_logger
  - from core.crawling.natural_language_parser import NaturalLanguageCrawlParser, CrawlCommand
  - from ingest.crawl4ai_runner import Crawl4aiRunner, CrawlStrategy, CrawlStatus
  - from core.services.file_watcher_service import get_file_watcher_manager
  - from suggestions.related.local_suggester import LocalSuggester
  - from suggestions.connections.local_connector import LocalConnector
  - from suggestions.organization.local_recommender import LocalRecommender
  - from suggestions.gaps.local_detector import LocalDetector
  - from agents.websearch_agent import WebSearchAgent
  - from models.memory import Memory
  - from models.memory import Memory
  - from models.memory import Memory
- External imports:
  - import asyncio
  - import json
  - import logging
  - import traceback
  - from datetime import datetime, timedelta
  - ... and 1 more

### src/mcp/tools.py
- Module: src.mcp.tools
- Entry point: False
- Functions: 17
- Classes: 3
- Total imports: 4
- Internal imports: 1
- External imports: 3
- Internal imports:
  - from mcp.types import Tool
- External imports:
  - import json
  - from datetime import datetime
  - from typing import Any, Dict, List, Optional

### src/mcp/transport.py
- Module: src.mcp.transport
- Entry point: False
- Functions: 4
- Classes: 4
- Total imports: 12
- Internal imports: 0
- External imports: 12
- External imports:
  - import asyncio
  - import json
  - import logging
  - from abc import ABC, abstractmethod
  - from typing import Any, Callable, Dict, Optional
  - ... and 7 more

### src/memory/__init__.py
- Module: src.memory
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 2
- Internal imports: 2
- External imports: 0
- Internal imports:
  - from pgvector_handler import PgVectorHandler, MemoryChunk, ChunkStatus, ChunkSearchResult, ChunkStats, SearchConfig
  - from neo4j_linker import Neo4jLinker, NodeType, RelationType, TrustLevel, GraphNode, GraphRelationship

### src/memory/neo4j_linker.py
- Module: src.memory.neo4j_linker
- Entry point: False
- Functions: 2
- Classes: 6
- Total imports: 16
- Internal imports: 3
- External imports: 13
- Internal imports:
  - from agents.websearch_agent import WebSearchResult
  - from core.utils.config import settings
  - from agents.websearch_agent import WebSearchResult, ContentExtractor, ExtractionQuality
- External imports:
  - import asyncio
  - import json
  - import hashlib
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any, Union, Tuple, Set
  - ... and 8 more

### src/memory/pgvector_handler.py
- Module: src.memory.pgvector_handler
- Entry point: False
- Functions: 5
- Classes: 7
- Total imports: 16
- Internal imports: 1
- External imports: 15
- Internal imports:
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import json
  - import uuid
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any, Union, Tuple
  - ... and 10 more

### src/migrations/__init__.py
- Module: src.migrations
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/suggestions/__init__.py
- Module: src.suggestions
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/validators/__init__.py
- Module: src.validators
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from memory_confidence import MemoryConfidenceAgent, ConfidenceLevel, ConfidenceFactor, ConfidenceResult, FactorType, ContentMetrics, DomainReputation

### src/validators/memory_confidence.py
- Module: src.validators.memory_confidence
- Entry point: False
- Functions: 17
- Classes: 7
- Total imports: 15
- Internal imports: 2
- External imports: 13
- Internal imports:
  - from core.utils.config import settings
  - from memory.neo4j_linker import Neo4jLinker
- External imports:
  - import asyncio
  - import hashlib
  - import re
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any, Union, Tuple
  - ... and 8 more

### tests/integration/test_end_to_end_workflows.py
- Module: tests.integration.test_end_to_end_workflows
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 12
- Internal imports: 5
- External imports: 7
- Internal imports:
  - from mcp.types import CallToolRequest, CallToolResult, TextContent
  - from src.mcp.server import TyraMemoryServer
  - from src.core.memory.manager import MemoryManager
  - from src.core.rag.hallucination_detector import HallucinationDetector
  - from src.core.analytics.performance_tracker import PerformanceTracker
- External imports:
  - import asyncio
  - import json
  - import pytest
  - import uuid
  - from datetime import datetime, timedelta
  - ... and 2 more

### tests/integration/test_graph_integration.py
- Module: tests.integration.test_graph_integration
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 9
- Internal imports: 4
- External imports: 5
- Internal imports:
  - from src.core.graph.graphiti_integration import GraphitiManager
  - from src.core.graph.neo4j_client import Neo4jClient
  - from src.core.providers.graph_engines.neo4j import Neo4jEngine
  - from src.core.interfaces.graph_engine import Entity, Relationship, RelationshipType
- External imports:
  - import pytest
  - import uuid
  - from datetime import datetime, timedelta
  - from typing import Any, Dict, List
  - from unittest.mock import AsyncMock, MagicMock, patch

### tests/integration/test_provider_integration.py
- Module: tests.integration.test_provider_integration
- Entry point: False
- Functions: 3
- Classes: 4
- Total imports: 9
- Internal imports: 5
- External imports: 4
- Internal imports:
  - from src.core.utils.registry import ProviderRegistry, ProviderType
  - from src.core.interfaces.embeddings import EmbeddingProvider
  - from src.core.interfaces.vector_store import VectorStore
  - from src.core.interfaces.graph_engine import GraphEngine
  - from src.core.interfaces.reranker import Reranker
- External imports:
  - import asyncio
  - import pytest
  - from unittest.mock import AsyncMock, MagicMock, patch
  - from typing import Any, Dict, List

### tests/performance/__init__.py
- Module: tests.performance
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### tests/stress/__init__.py
- Module: tests.stress
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### scripts/migrations/config/001_initial_config.py
- Module: scripts.migrations.config.001_initial_config
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 1
- Internal imports: 0
- External imports: 1
- External imports:
  - from typing import Dict, Any

### scripts/migrations/config/002_add_observability.py
- Module: scripts.migrations.config.002_add_observability
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 1
- Internal imports: 0
- External imports: 1
- External imports:
  - from typing import Dict, Any

### scripts/migrations/config/003_add_self_learning.py
- Module: scripts.migrations.config.003_add_self_learning
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 1
- Internal imports: 0
- External imports: 1
- External imports:
  - from typing import Dict, Any

### scripts/migrations/config/004_add_trading_safety.py
- Module: scripts.migrations.config.004_add_trading_safety
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 1
- Internal imports: 0
- External imports: 1
- External imports:
  - from typing import Dict, Any

### scripts/migrations/config/__init__.py
- Module: scripts.migrations.config
- Entry point: False
- Functions: 2
- Classes: 0
- Total imports: 4
- Internal imports: 0
- External imports: 4
- External imports:
  - import importlib
  - import os
  - from pathlib import Path
  - from typing import List, Type

### src/api/middleware/__init__.py
- Module: src.api.middleware
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/api/middleware/auth.py
- Module: src.api.middleware.auth
- Entry point: False
- Functions: 9
- Classes: 3
- Total imports: 20
- Internal imports: 2
- External imports: 18
- Internal imports:
  - from core.utils.config import get_settings
  - from core.utils.logger import get_logger
- External imports:
  - import hashlib
  - import hmac
  - import time
  - from typing import Optional, Set
  - from fastapi import HTTPException, Request, status
  - ... and 13 more

### src/api/middleware/error_handler.py
- Module: src.api.middleware.error_handler
- Entry point: False
- Functions: 12
- Classes: 5
- Total imports: 10
- Internal imports: 2
- External imports: 8
- Internal imports:
  - from core.utils.config import get_settings
  - from core.utils.logger import get_logger
- External imports:
  - import time
  - import traceback
  - import uuid
  - from typing import Any, Dict, Optional
  - from fastapi import HTTPException, Request
  - ... and 3 more

### src/api/middleware/rate_limit.py
- Module: src.api.middleware.rate_limit
- Entry point: False
- Functions: 8
- Classes: 2
- Total imports: 11
- Internal imports: 4
- External imports: 7
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import get_settings
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - import hashlib
  - import json
  - import time
  - from typing import Any, Dict, Optional
  - from fastapi import HTTPException, Request, status
  - ... and 2 more

### src/api/routes/__init__.py
- Module: src.api.routes
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 21
- Internal imports: 21
- External imports: 0
- Internal imports:
  - from admin import router
  - from analytics import router
  - from chat import router
  - from crawling import router
  - from embeddings import router
  - from file_watcher import router
  - from graph import router
  - from health import router
  - from ingestion import router
  - from memory import router
  - from observability import router
  - from performance import router
  - from personalization import router
  - from prediction import router
  - from rag import router
  - from search import router
  - from security import router
  - from suggestions import router
  - from synthesis import router
  - from trading_data import router
  - from webhooks import router

### src/api/routes/admin.py
- Module: src.api.routes.admin
- Entry point: False
- Functions: 3
- Classes: 9
- Total imports: 41
- Internal imports: 10
- External imports: 31
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.ingestion.job_tracker import JobTracker, JobStatus, JobType, get_job_tracker
  - from core.memory.manager import MemoryManager
  - from core.observability.telemetry import get_telemetry
  - from core.schemas.ingestion import IngestionProgress
  - from core.utils.config import get_settings, reload_config
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
  - from core.utils.registry import ProviderType, get_provider
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - from datetime import datetime, timedelta
  - from enum import Enum
  - from typing import Any, Dict, List, Optional
  - from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
  - from pydantic import BaseModel, Field
  - ... and 26 more

### src/api/routes/analytics.py
- Module: src.api.routes.analytics
- Entry point: False
- Functions: 0
- Classes: 5
- Total imports: 11
- Internal imports: 6
- External imports: 5
- Internal imports:
  - from src.core.analytics.performance_tracker import PerformanceTracker, MetricType
  - from src.core.adaptation.memory_health import MemoryHealthManager
  - from src.core.adaptation.config_optimizer import ConfigOptimizer
  - from src.core.utils.auth import get_current_user_optional
  - from src.core.utils.logger import get_logger
  - from src.core.utils.rate_limiter import RateLimiter
- External imports:
  - import asyncio
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any
  - from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
  - from pydantic import BaseModel, Field

### src/api/routes/chat.py
- Module: src.api.routes.chat
- Entry point: False
- Functions: 4
- Classes: 9
- Total imports: 20
- Internal imports: 6
- External imports: 14
- Internal imports:
  - from core.memory.manager import MemoryManager
  - from core.rag.reranker import Reranker
  - from core.rag.hallucination_detector import HallucinationDetector
  - from core.search.searcher import Searcher
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - import asyncio
  - import json
  - from datetime import datetime
  - from enum import Enum
  - from typing import Any, AsyncGenerator, Dict, List, Optional
  - ... and 9 more

### src/api/routes/crawling.py
- Module: src.api.routes.crawling
- Entry point: False
- Functions: 0
- Classes: 5
- Total imports: 11
- Internal imports: 7
- External imports: 4
- Internal imports:
  - from core.crawling.natural_language_parser import NaturalLanguageCrawlParser
  - from ingest.crawl4ai_runner import Crawl4aiRunner, CrawlStrategy, CrawlStatus
  - from core.memory.manager import MemoryManager
  - from core.utils.simple_logger import get_logger
  - from middleware.auth import verify_api_key
  - from middleware.rate_limit import rate_limit
  - from ingest.crawl4ai_runner import DomainPolicy
- External imports:
  - from datetime import datetime
  - from typing import Dict, List, Optional, Any
  - from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
  - from pydantic import BaseModel, Field

### src/api/routes/embeddings.py
- Module: src.api.routes.embeddings
- Entry point: False
- Functions: 2
- Classes: 11
- Total imports: 14
- Internal imports: 8
- External imports: 6
- Internal imports:
  - from core.embeddings.contextual import ContextualEmbedder, SessionContext, ContextualResult
  - from core.embeddings.multi_perspective import MultiPerspectiveEmbedder, PerspectiveType, EmbeddingContext
  - from core.embeddings.fine_tuning import DynamicFineTuner, FineTuningConfig, FineTuningResult
  - from core.embeddings.fusion import EmbeddingFusionSystem, FusionConfig, FusionResult
  - from core.embeddings.embedder import Embedder
  - from core.cache.redis_cache import RedisCache
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - import asyncio
  - import time
  - from typing import Dict, List, Optional, Any, Union
  - from datetime import datetime
  - from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
  - ... and 1 more

### src/api/routes/file_watcher.py
- Module: src.api.routes.file_watcher
- Entry point: False
- Functions: 0
- Classes: 3
- Total imports: 9
- Internal imports: 5
- External imports: 4
- Internal imports:
  - from core.services.file_watcher_service import get_file_watcher_manager
  - from core.utils.simple_logger import get_logger
  - from middleware.auth import verify_api_key
  - from middleware.rate_limit import rate_limit
  - from core.utils.simple_config import get_settings
- External imports:
  - from datetime import datetime
  - from typing import Dict, Any, Optional
  - from fastapi import APIRouter, Depends, HTTPException, Query
  - from pydantic import BaseModel, Field

### src/api/routes/graph.py
- Module: src.api.routes.graph
- Entry point: False
- Functions: 7
- Classes: 15
- Total imports: 14
- Internal imports: 5
- External imports: 9
- Internal imports:
  - from core.graph.graphiti_integration import GraphitiManager
  - from core.graph.neo4j_client import Neo4jClient
  - from core.graph.causal_inference import CausalInferenceEngine, CausalRelationType, CausalInferenceMethod
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - from datetime import datetime
  - from enum import Enum
  - from typing import Any, Dict, List, Optional, Set
  - from fastapi import APIRouter, Depends, HTTPException, Query
  - from pydantic import BaseModel, Field
  - ... and 4 more

### src/api/routes/health.py
- Module: src.api.routes.health
- Entry point: False
- Functions: 0
- Classes: 6
- Total imports: 10
- Internal imports: 2
- External imports: 8
- Internal imports:
  - from core.utils.logger import get_logger
  - from app import get_memory_manager
- External imports:
  - from datetime import datetime
  - from enum import Enum
  - from typing import Any, Dict, List
  - from fastapi import APIRouter, Depends, HTTPException
  - from pydantic import BaseModel, Field
  - ... and 3 more

### src/api/routes/ingestion.py
- Module: src.api.routes.ingestion
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 20
- Internal imports: 10
- External imports: 10
- Internal imports:
  - from core.ingestion.document_processor import DocumentProcessor
  - from core.ingestion.file_loaders import get_file_loader
  - from core.ingestion.job_tracker import JobTracker, JobType, JobStatus, get_job_tracker
  - from core.memory.manager import MemoryManager
  - from core.observability import get_tracer
  - from core.schemas.ingestion import BatchIngestRequest, BatchIngestResponse, IngestRequest, IngestResponse, IngestionCapabilities, IngestionProgress, IngestionWarning, SupportedFormats
  - from core.utils.config import get_settings
  - from core.utils.logger import get_logger
  - from core.memory.manager import MemoryManager
  - from core.schemas.ingestion import DocumentMetadata
- External imports:
  - import asyncio
  - import base64
  - import io
  - import time
  - import uuid
  - ... and 5 more

### src/api/routes/memory.py
- Module: src.api.routes.memory
- Entry point: False
- Functions: 0
- Classes: 7
- Total imports: 8
- Internal imports: 3
- External imports: 5
- Internal imports:
  - from core.memory.manager import MemoryManager
  - from core.utils.logger import get_logger
  - from app import get_memory_manager, get_request_context
- External imports:
  - import uuid
  - from datetime import datetime
  - from typing import Any, Dict, List, Optional
  - from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
  - from pydantic import BaseModel, Field

### src/api/routes/observability.py
- Module: src.api.routes.observability
- Entry point: False
- Functions: 0
- Classes: 11
- Total imports: 13
- Internal imports: 7
- External imports: 6
- Internal imports:
  - from core.observability.dashboard import MemoryQualityDashboard, QualityMetric, AlertConfig
  - from core.observability.telemetry import TelemetryCollector, TelemetryData
  - from core.observability.metrics import MetricsCollector, SystemMetrics
  - from core.observability.tracing import DistributedTracer, TraceContext
  - from core.cache.redis_cache import RedisCache
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - import asyncio
  - import time
  - from typing import Dict, List, Optional, Any
  - from datetime import datetime, timedelta
  - from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
  - ... and 1 more

### src/api/routes/performance.py
- Module: src.api.routes.performance
- Entry point: False
- Functions: 0
- Classes: 9
- Total imports: 12
- Internal imports: 5
- External imports: 7
- Internal imports:
  - from infrastructure.scaling.auto_scaler import AutoScalingEngine, ScalingStrategy, MetricType
  - from core.optimization.query_optimizer import QueryOptimizer, QueryOptimization, IndexRecommendation
  - from core.cache.redis_cache import RedisCache
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - import asyncio
  - import time
  - from typing import Dict, List, Optional, Any
  - from datetime import datetime, timedelta
  - from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
  - ... and 2 more

### src/api/routes/personalization.py
- Module: src.api.routes.personalization
- Entry point: False
- Functions: 0
- Classes: 11
- Total imports: 11
- Internal imports: 5
- External imports: 6
- Internal imports:
  - from core.personalization.preferences.user_preference_engine import UserPreferenceEngine, PreferenceType, UserProfile
  - from core.personalization.confidence.adaptive_confidence import AdaptiveConfidenceManager, ConfidenceContext
  - from core.cache.redis_cache import RedisCache
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - import asyncio
  - import time
  - from typing import Dict, List, Optional, Any
  - from datetime import datetime, timedelta
  - from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
  - ... and 1 more

### src/api/routes/prediction.py
- Module: src.api.routes.prediction
- Entry point: False
- Functions: 0
- Classes: 9
- Total imports: 13
- Internal imports: 8
- External imports: 5
- Internal imports:
  - from core.prediction.usage_analyzer import UsageAnalyzer, AnalysisConfig, UsagePattern
  - from core.prediction.auto_archiver import AutoArchiver, ArchivingPolicy, ArchivingResult
  - from core.prediction.preloader import PredictivePreloader, PreloadingConfig, PreloadingResult
  - from core.prediction.lifecycle import LifecycleOptimizer, MemoryStage, LifecycleAnalysisResult
  - from core.memory.manager import MemoryManager
  - from core.cache.redis_cache import RedisCache
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - import asyncio
  - from typing import Dict, List, Optional, Any
  - from datetime import datetime, timedelta
  - from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
  - from pydantic import BaseModel, Field

### src/api/routes/rag.py
- Module: src.api.routes.rag
- Entry point: False
- Functions: 4
- Classes: 11
- Total imports: 18
- Internal imports: 10
- External imports: 8
- Internal imports:
  - from core.rag.pipeline_manager import IntegratedRAGPipeline, RAGPipelineResult, PipelineConfig, PipelineMode, PipelineStage
  - from core.rag.hallucination_detector import HallucinationDetector
  - from core.rag.reranker import Reranker
  - from core.memory.manager import MemoryManager
  - from core.embeddings.embedder import Embedder
  - from core.graph.neo4j_client import Neo4jClient
  - from core.cache.redis_cache import RedisCache
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
  - from core.graph.neo4j_client import get_neo4j_client
- External imports:
  - from enum import Enum
  - from typing import Any, Dict, List, Optional
  - from fastapi import APIRouter, Depends, HTTPException, Query
  - from pydantic import BaseModel, Field
  - from datetime import datetime, timedelta
  - ... and 3 more

### src/api/routes/search.py
- Module: src.api.routes.search
- Entry point: False
- Functions: 1
- Classes: 7
- Total imports: 14
- Internal imports: 4
- External imports: 10
- Internal imports:
  - from core.memory.models import Memory
  - from core.search.searcher import Searcher
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - from datetime import datetime
  - from enum import Enum
  - from typing import Any, Dict, List, Optional
  - from fastapi import APIRouter, Depends, HTTPException, Query
  - from pydantic import BaseModel, Field
  - ... and 5 more

### src/api/routes/security.py
- Module: src.api.routes.security
- Entry point: False
- Functions: 0
- Classes: 8
- Total imports: 13
- Internal imports: 6
- External imports: 7
- Internal imports:
  - from core.security.rbac import RBACManager, Role, User, Permission, AccessDecision
  - from core.security.encryption import MemoryEncryption, EncryptionConfig, EncryptionResult
  - from core.security.audit import AuditLogger, AuditEvent, SecurityEvent, ComplianceReport
  - from core.cache.redis_cache import RedisCache
  - from core.utils.logger import get_logger
  - from core.utils.registry import ProviderType, get_provider
- External imports:
  - import asyncio
  - import time
  - from typing import Dict, List, Optional, Any
  - from datetime import datetime, timedelta
  - from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security
  - ... and 2 more

### src/api/routes/suggestions.py
- Module: src.api.routes.suggestions
- Entry point: False
- Functions: 0
- Classes: 8
- Total imports: 13
- Internal imports: 7
- External imports: 6
- Internal imports:
  - from suggestions.related.local_suggester import LocalSuggester, SuggestionType, RelevanceScore
  - from suggestions.connections.local_connector import LocalConnector
  - from suggestions.organization.local_recommender import LocalRecommender
  - from suggestions.gaps.local_detector import LocalDetector
  - from core.memory.manager import MemoryManager
  - from core.providers.embeddings.embedder import Embedder
  - from core.utils.logger import get_logger
- External imports:
  - import asyncio
  - from datetime import datetime, timedelta
  - from typing import Any, Dict, List, Optional, Union
  - from enum import Enum
  - from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
  - ... and 1 more

### src/api/routes/synthesis.py
- Module: src.api.routes.synthesis
- Entry point: False
- Functions: 0
- Classes: 8
- Total imports: 13
- Internal imports: 7
- External imports: 6
- Internal imports:
  - from core.synthesis import DeduplicationEngine, SummarizationEngine, PatternDetector, TemporalAnalyzer, SummarizationType, DuplicateType, MergeStrategy, PatternType
  - from core.memory.manager import MemoryManager, MemorySearchRequest
  - from core.clients.vllm_client import VLLMClient
  - from core.rag.hallucination_detector import HallucinationDetector
  - from core.providers.embeddings.embedder import Embedder
  - from models.memory import Memory
  - from core.utils.logger import get_logger
- External imports:
  - import asyncio
  - from datetime import datetime, timedelta
  - from typing import Any, Dict, List, Optional, Union
  - from enum import Enum
  - from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
  - ... and 1 more

### src/api/routes/trading_data.py
- Module: src.api.routes.trading_data
- Entry point: False
- Functions: 0
- Classes: 9
- Total imports: 12
- Internal imports: 3
- External imports: 9
- Internal imports:
  - from core.memory.manager import MemoryManager
  - from core.utils.logger import get_logger
  - from app import get_memory_manager
- External imports:
  - import asyncio
  - import json
  - import uuid
  - from datetime import datetime, timedelta
  - from decimal import Decimal
  - ... and 4 more

### src/api/routes/webhooks.py
- Module: src.api.routes.webhooks
- Entry point: False
- Functions: 0
- Classes: 5
- Total imports: 12
- Internal imports: 4
- External imports: 8
- Internal imports:
  - from src.api.dependencies import get_memory_manager
  - from src.core.agents.agent_logger import agent_log_context, get_agent_logger
  - from src.core.agents.session_manager import get_session_manager
  - from src.core.memory.manager import MemoryManager
- External imports:
  - import json
  - import uuid
  - from datetime import datetime
  - from typing import Any, Dict, List, Optional
  - from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
  - ... and 3 more

### src/api/websocket/__init__.py
- Module: src.api.websocket
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 3
- Internal imports: 3
- External imports: 0
- Internal imports:
  - from server import WebSocketServer, ConnectionManager, WebSocketConnection, ConnectionState, MessageType
  - from memory_stream import MemoryStreamManager, MemoryEvent, MemoryEventType, MemorySubscription, StreamFilter
  - from search_stream import SearchStreamManager, SearchEvent, SearchEventType, QuerySuggestion, SearchState

### src/api/websocket/memory_stream.py
- Module: src.api.websocket.memory_stream
- Entry point: False
- Functions: 5
- Classes: 5
- Total imports: 14
- Internal imports: 3
- External imports: 11
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from server import ConnectionManager, MessageType
- External imports:
  - import asyncio
  - import json
  - import time
  - from typing import Dict, List, Optional, Set, Any, Callable, Union
  - from datetime import datetime, timedelta
  - ... and 6 more

### src/api/websocket/search_stream.py
- Module: src.api.websocket.search_stream
- Entry point: False
- Functions: 14
- Classes: 8
- Total imports: 19
- Internal imports: 4
- External imports: 15
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.embeddings.embedder import Embedder
  - from server import ConnectionManager, MessageType
- External imports:
  - import asyncio
  - import json
  - import time
  - from typing import Dict, List, Optional, Set, Any, Callable, Union, Tuple
  - from datetime import datetime, timedelta
  - ... and 10 more

### src/api/websocket/server.py
- Module: src.api.websocket.server
- Entry point: False
- Functions: 8
- Classes: 6
- Total imports: 20
- Internal imports: 2
- External imports: 18
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import json
  - import time
  - import uuid
  - from typing import Dict, List, Optional, Set, Any, Callable, Union
  - ... and 13 more

### src/core/adaptation/__init__.py
- Module: src.core.adaptation
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from learning_engine import AdaptationExperiment, AdaptationParameter, AdaptationStatus, AdaptationStrategy, AdaptationType, BayesianOptimizationStrategy, GradientDescentStrategy, LearningEngine, LearningInsight, RandomSearchStrategy

### src/core/adaptation/ab_testing.py
- Module: src.core.adaptation.ab_testing
- Entry point: False
- Functions: 6
- Classes: 8
- Total imports: 28
- Internal imports: 10
- External imports: 18
- Internal imports:
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager, get_redis_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
- External imports:
  - import asyncio
  - import hashlib
  - import json
  - import logging
  - import random
  - ... and 13 more

### src/core/adaptation/learning_engine.py
- Module: src.core.adaptation.learning_engine
- Entry point: False
- Functions: 15
- Classes: 10
- Total imports: 14
- Internal imports: 3
- External imports: 11
- Internal imports:
  - from analytics.performance_tracker import MetricType, OptimizationRecommendation, PerformanceTracker
  - from utils.config import get_settings
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import json
  - import random
  - import time
  - from abc import ABC, abstractmethod
  - ... and 6 more

### src/core/adaptation/memory_health.py
- Module: src.core.adaptation.memory_health
- Entry point: False
- Functions: 7
- Classes: 4
- Total imports: 8
- Internal imports: 0
- External imports: 8
- External imports:
  - import asyncio
  - import logging
  - from collections import defaultdict
  - from dataclasses import dataclass, field
  - from datetime import datetime, timedelta
  - ... and 3 more

### src/core/adaptation/prompt_evolution.py
- Module: src.core.adaptation.prompt_evolution
- Entry point: False
- Functions: 13
- Classes: 4
- Total imports: 24
- Internal imports: 7
- External imports: 17
- Internal imports:
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
- External imports:
  - import asyncio
  - import json
  - import logging
  - import re
  - from collections import defaultdict
  - ... and 12 more

### src/core/adaptation/self_training_scheduler.py
- Module: src.core.adaptation.self_training_scheduler
- Entry point: False
- Functions: 5
- Classes: 5
- Total imports: 42
- Internal imports: 29
- External imports: 13
- Internal imports:
  - from memory_health import MemoryHealthManager
  - from config_optimizer import ConfigOptimizer
  - from ab_testing import ABTestingFramework
  - from prompt_evolution import PromptEvolutionEngine
  - from analytics.performance_tracker import PerformanceTracker
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.config import update_config
  - from utils.database import get_postgres_manager
  - from utils.config import update_config
  - from utils.database import get_postgres_manager
  - from analytics.performance_tracker import get_performance_tracker
  - from utils.database import get_redis_manager, get_postgres_manager
  - from utils.config import update_config
  - from analytics.performance_tracker import get_performance_tracker
  - from analytics.performance_tracker import get_performance_tracker
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from analytics.performance_tracker import get_performance_tracker
  - from utils.database import get_postgres_manager
  - from utils.database import get_redis_manager
  - from cache.redis_cache import RedisCache
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
  - from utils.database import get_postgres_manager
- External imports:
  - import asyncio
  - import json
  - import logging
  - from datetime import datetime, timedelta
  - from dataclasses import dataclass, field
  - ... and 8 more

### src/core/agents/__init__.py
- Module: src.core.agents
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from session_manager import AgentSession, AgentSessionManager, create_agent_session, get_agent_session, get_session_manager

### src/core/agents/agent_logger.py
- Module: src.core.agents.agent_logger
- Entry point: False
- Functions: 19
- Classes: 3
- Total imports: 5
- Internal imports: 1
- External imports: 4
- Internal imports:
  - from src.core.utils.logger import get_logger
- External imports:
  - import contextvars
  - import logging
  - from datetime import datetime
  - from typing import Any, Dict, Optional

### src/core/agents/claude_integration.py
- Module: src.core.agents.claude_integration
- Entry point: False
- Functions: 1
- Classes: 1
- Total imports: 10
- Internal imports: 5
- External imports: 5
- Internal imports:
  - from src.clients.memory_client import MemoryClient
  - from src.core.agents.agent_logger import agent_log_context, get_agent_logger
  - from src.core.agents.session_manager import AgentSessionManager, create_agent_session
  - from src.core.memory.manager import MemoryManager
  - from src.core.agents import get_session_manager
- External imports:
  - import asyncio
  - import json
  - from datetime import datetime
  - from typing import Any, Dict, List, Optional, Tuple
  - import asyncio

### src/core/agents/session_manager.py
- Module: src.core.agents.session_manager
- Entry point: False
- Functions: 7
- Classes: 2
- Total imports: 10
- Internal imports: 3
- External imports: 7
- Internal imports:
  - from src.core.cache.redis_cache import CacheLevel, RedisCache
  - from src.core.utils.config import load_config
  - from src.core.utils.logger import get_logger
- External imports:
  - import asyncio
  - import time
  - import uuid
  - from contextlib import asynccontextmanager
  - from dataclasses import dataclass, field
  - ... and 2 more

### src/core/ai/cross_component_validation.py
- Module: src.core.ai.cross_component_validation
- Entry point: False
- Functions: 8
- Classes: 8
- Total imports: 17
- Internal imports: 6
- External imports: 11
- Internal imports:
  - from clients.vllm_client import VLLMClient
  - from embeddings.embedder import Embedder
  - from utils.config import settings
  - from structured_operations import StructuredOperationsManager
  - from logits_processors import LogitsProcessorManager
  - from mcp_response_validator import MCPResponseValidator
- External imports:
  - import asyncio
  - import json
  - import math
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any, Union, Tuple, Type
  - ... and 6 more

### src/core/ai/logits_processors.py
- Module: src.core.ai.logits_processors
- Entry point: False
- Functions: 27
- Classes: 9
- Total imports: 14
- Internal imports: 3
- External imports: 11
- Internal imports:
  - from clients.vllm_client import VLLMClient
  - from embeddings.embedder import Embedder
  - from utils.config import settings
- External imports:
  - import asyncio
  - import math
  - import re
  - from typing import Dict, List, Optional, Any, Union, Tuple, Callable
  - from dataclasses import dataclass
  - ... and 6 more

### src/core/ai/mcp_response_validator.py
- Module: src.core.ai.mcp_response_validator
- Entry point: False
- Functions: 7
- Classes: 8
- Total imports: 15
- Internal imports: 4
- External imports: 11
- Internal imports:
  - from mcp.types import CallToolResult, TextContent, ImageContent, EmbeddedResource
  - from clients.vllm_client import VLLMClient
  - from utils.config import settings
  - from mcp.types import CallToolResult, TextContent
- External imports:
  - import asyncio
  - import json
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any, Union, Tuple, Type
  - from dataclasses import dataclass
  - ... and 6 more

### src/core/ai/structured_operations.py
- Module: src.core.ai.structured_operations
- Entry point: False
- Functions: 12
- Classes: 15
- Total imports: 16
- Internal imports: 3
- External imports: 13
- Internal imports:
  - from clients.vllm_client import VLLMClient, ChatMessage, Role
  - from embeddings.embedder import Embedder
  - from utils.config import settings
- External imports:
  - import asyncio
  - import json
  - import re
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any, Union, Tuple, Set
  - ... and 8 more

### src/core/analytics/__init__.py
- Module: src.core.analytics
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 3
- Internal imports: 3
- External imports: 0
- Internal imports:
  - from config_optimizer import ConfigParameter, ConfigurationOptimizer, OptimizationExperiment, OptimizationResult, OptimizationStrategy, get_config_optimizer
  - from memory_health import HealthCategory, HealthCheck, HealthMetric, HealthStatus, MemoryHealthManager, SystemSnapshot, get_health_manager
  - from performance_tracker import MetricType, PerformanceBaseline, PerformanceEvent, PerformancePattern, PerformanceTracker, get_performance_tracker, record_operation_latency, track_performance

### src/core/analytics/config_optimizer.py
- Module: src.core.analytics.config_optimizer
- Entry point: False
- Functions: 15
- Classes: 6
- Total imports: 12
- Internal imports: 2
- External imports: 10
- Internal imports:
  - from memory_health import HealthStatus, get_health_manager
  - from performance_tracker import MetricType, get_performance_tracker
- External imports:
  - import asyncio
  - import json
  - import logging
  - from dataclasses import asdict, dataclass, field
  - from datetime import datetime, timedelta
  - ... and 5 more

### src/core/analytics/memory_health.py
- Module: src.core.analytics.memory_health
- Entry point: False
- Functions: 21
- Classes: 6
- Total imports: 24
- Internal imports: 6
- External imports: 18
- Internal imports:
  - from core.utils.database import get_postgres_manager
  - from core.cache.redis_cache import get_redis_client
  - from core.analytics.performance_tracker import get_performance_tracker
  - from core.analytics.performance_tracker import get_performance_tracker
  - from core.cache.redis_cache import get_redis_client
  - from core.utils.database import get_postgres_manager
- External imports:
  - import asyncio
  - import logging
  - import time
  - from dataclasses import dataclass, field
  - from datetime import datetime, timedelta
  - ... and 13 more

### src/core/analytics/performance_tracker.py
- Module: src.core.analytics.performance_tracker
- Entry point: False
- Functions: 22
- Classes: 6
- Total imports: 11
- Internal imports: 0
- External imports: 11
- External imports:
  - import asyncio
  - import json
  - import logging
  - import statistics
  - from collections import defaultdict, deque
  - ... and 6 more

### src/core/cache/__init__.py
- Module: src.core.cache
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from redis_cache import CacheEntry, CacheLevel, CacheMetrics, RedisCache

### src/core/cache/cache_manager.py
- Module: src.core.cache.cache_manager
- Entry point: False
- Functions: 7
- Classes: 2
- Total imports: 13
- Internal imports: 3
- External imports: 10
- Internal imports:
  - from utils.logger import get_logger
  - from utils.circuit_breaker import circuit_breaker
  - from observability.tracing import trace_method
- External imports:
  - import asyncio
  - import hashlib
  - import json
  - import pickle
  - from datetime import datetime, timedelta
  - ... and 5 more

### src/core/cache/redis_cache.py
- Module: src.core.cache.redis_cache
- Entry point: False
- Functions: 9
- Classes: 4
- Total imports: 13
- Internal imports: 3
- External imports: 10
- Internal imports:
  - from utils.config import get_settings
  - from utils.database import RedisManager
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import json
  - import pickle
  - import time
  - import zlib
  - ... and 5 more

### src/core/clients/__init__.py
- Module: src.core.clients
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from vllm_client import VLLMClient

### src/core/clients/vllm_client.py
- Module: src.core.clients.vllm_client
- Entry point: False
- Functions: 6
- Classes: 6
- Total imports: 12
- Internal imports: 2
- External imports: 10
- Internal imports:
  - from utils.config import settings
  - from utils.circuit_breaker import CircuitBreaker
- External imports:
  - import asyncio
  - import json
  - from typing import List, Dict, Optional, Union, Any, AsyncIterator
  - from dataclasses import dataclass
  - from enum import Enum
  - ... and 5 more

### src/core/crawling/__init__.py
- Module: src.core.crawling
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from natural_language_parser import NaturalLanguageCrawlParser, CrawlCommand, ParsedIntent

### src/core/crawling/natural_language_parser.py
- Module: src.core.crawling.natural_language_parser
- Entry point: False
- Functions: 12
- Classes: 3
- Total imports: 6
- Internal imports: 2
- External imports: 4
- Internal imports:
  - from ingest.crawl4ai_runner import CrawlStrategy, ContentType
  - from utils.simple_logger import get_logger
- External imports:
  - import re
  - from typing import Dict, List, Optional, Tuple, Any
  - from dataclasses import dataclass
  - from enum import Enum

### src/core/embeddings/__init__.py
- Module: src.core.embeddings
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 6
- Internal imports: 6
- External imports: 0
- Internal imports:
  - from manager import EmbeddingManager
  - from models import EmbeddingRequest, EmbeddingResponse
  - from contextual import SessionAwareEmbedder, ContextualEmbedding, SessionContext, ContextType, FusionStrategy, ContextDecayModel, ContextVectorFusion
  - from multi_perspective import MultiPerspectiveEmbedder, MultiPerspectiveResult, PerspectiveEmbedding, PerspectiveModel, PerspectiveType, RoutingStrategy, ConfidenceMethod, PerspectiveRouter, ConfidenceEstimator
  - from fine_tuning import DynamicFineTuner, FineTuningConfig, FineTuningStrategy, LearningObjective, TrainingExample, ModelVersion, ModelVersionManager, OnlineLearningOptimizer
  - from fusion import EmbeddingFusionEngine, FusionResult, EmbeddingSource, FusionConfig, FusionStrategy, QualityMetric, QualityAssessor, AdaptiveFusionOptimizer, AttentionFusionNetwork

### src/core/embeddings/contextual.py
- Module: src.core.embeddings.contextual
- Entry point: False
- Functions: 10
- Classes: 9
- Total imports: 22
- Internal imports: 3
- External imports: 19
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Deque
  - from datetime import datetime, timedelta
  - ... and 14 more

### src/core/embeddings/fine_tuning.py
- Module: src.core.embeddings.fine_tuning
- Entry point: False
- Functions: 16
- Classes: 14
- Total imports: 29
- Internal imports: 3
- External imports: 26
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 21 more

### src/core/embeddings/fusion.py
- Module: src.core.embeddings.fusion
- Entry point: False
- Functions: 11
- Classes: 10
- Total imports: 30
- Internal imports: 5
- External imports: 25
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
  - from contextual import SessionAwareEmbedder, ContextualEmbedding
  - from multi_perspective import MultiPerspectiveEmbedder, MultiPerspectiveResult
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 20 more

### src/core/embeddings/manager.py
- Module: src.core.embeddings.manager
- Entry point: False
- Functions: 1
- Classes: 1
- Total imports: 9
- Internal imports: 4
- External imports: 5
- Internal imports:
  - from providers.embeddings.registry import get_embedding_provider
  - from cache.cache_manager import CacheManager
  - from utils.config import get_config
  - from models import EmbeddingRequest, EmbeddingResponse, BatchEmbeddingRequest, BatchEmbeddingResponse
- External imports:
  - import asyncio
  - import logging
  - import time
  - from typing import Any, Dict, List, Optional, Union
  - import numpy

### src/core/embeddings/models.py
- Module: src.core.embeddings.models
- Entry point: False
- Functions: 2
- Classes: 4
- Total imports: 4
- Internal imports: 0
- External imports: 4
- External imports:
  - from dataclasses import dataclass
  - from typing import Any, Dict, List, Optional
  - from datetime import datetime
  - import numpy

### src/core/embeddings/multi_perspective.py
- Module: src.core.embeddings.multi_perspective
- Entry point: False
- Functions: 7
- Classes: 9
- Total imports: 24
- Internal imports: 3
- External imports: 21
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 16 more

### src/core/events/__init__.py
- Module: src.core.events
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from trigger_system import TriggerEngine, Trigger, TriggerRule, TriggerAction, TriggerCondition, TriggerType, ConditionOperator, ActionType, ExecutionContext, TriggerResult

### src/core/events/trigger_system.py
- Module: src.core.events.trigger_system
- Entry point: False
- Functions: 16
- Classes: 11
- Total imports: 20
- Internal imports: 3
- External imports: 17
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import ast
  - import operator
  - import time
  - import json
  - ... and 12 more

### src/core/graph/__init__.py
- Module: src.core.graph
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 6
- Internal imports: 6
- External imports: 0
- Internal imports:
  - from graphiti_integration import GraphitiManager
  - from neo4j_client import Neo4jClient
  - from reasoning_engine import MultiHopReasoningEngine, ReasoningType, ReasoningStrategy, ConfidenceLevel, ReasoningNode, ReasoningEdge, ReasoningPath, ReasoningQuery, ReasoningResult, PathScorer, ExplanationGenerator
  - from causal_inference import CausalInferenceEngine, CausalRelationType, CausalInferenceMethod, CausalEvidence, CausalClaim, CausalGraph, GrangerCausalityAnalyzer, PearlCausalAnalyzer
  - from temporal_evolution import TemporalKnowledgeEvolutionEngine, EvolutionType, ChangeDetectionMethod, TemporalGranularity, TemporalSnapshot, EvolutionEvent, TemporalAlignment, ConceptDriftDetector, TemporalEmbeddingAligner, EvolutionPatternDetector
  - from recommender import GraphBasedRecommendationEngine, RecommendationType, RecommendationStrategy, RecommendationItem, RecommendationSet, UserProfile, ContentBasedRecommender, GraphEmbeddingRecommender, DiversityOptimizer

### src/core/graph/causal_inference.py
- Module: src.core.graph.causal_inference
- Entry point: False
- Functions: 19
- Classes: 9
- Total imports: 29
- Internal imports: 4
- External imports: 25
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
  - from reasoning_engine import ReasoningNode, ReasoningEdge, ReasoningPath
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 20 more

### src/core/graph/enhanced_graph_client.py
- Module: src.core.graph.enhanced_graph_client
- Entry point: False
- Functions: 1
- Classes: 1
- Total imports: 9
- Internal imports: 6
- External imports: 3
- Internal imports:
  - from utils.logger import get_logger
  - from utils.config import settings
  - from interfaces.graph_engine import GraphEngine, GraphEngineError
  - from observability.tracing import trace_method
  - from neo4j_client import Neo4jClient
  - from graphiti_integration import GraphitiManager
- External imports:
  - import asyncio
  - from datetime import datetime
  - from typing import Any, Dict, List, Optional, Tuple, Union

### src/core/graph/graphiti_integration.py
- Module: src.core.graph.graphiti_integration
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 18
- Internal imports: 4
- External imports: 14
- Internal imports:
  - from interfaces.graph_engine import GraphEngine, GraphEngineError
  - from utils.circuit_breaker import CircuitBreaker
  - from utils.logger import get_logger
  - from core.memory.manager import get_memory_manager
- External imports:
  - import asyncio
  - import logging
  - import time
  - import uuid
  - from datetime import datetime, timedelta
  - ... and 9 more

### src/core/graph/neo4j_client.py
- Module: src.core.graph.neo4j_client
- Entry point: False
- Functions: 1
- Classes: 1
- Total imports: 8
- Internal imports: 3
- External imports: 5
- Internal imports:
  - from interfaces.graph_engine import Entity, Relationship, GraphEngine
  - from providers.graph_engines.neo4j import Neo4jEngine
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import logging
  - import time
  - from datetime import datetime
  - from typing import Any, Dict, List, Optional

### src/core/graph/reasoning_engine.py
- Module: src.core.graph.reasoning_engine
- Entry point: False
- Functions: 15
- Classes: 11
- Total imports: 22
- Internal imports: 3
- External imports: 19
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 14 more

### src/core/graph/recommender.py
- Module: src.core.graph.recommender
- Entry point: False
- Functions: 17
- Classes: 10
- Total imports: 30
- Internal imports: 5
- External imports: 25
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
  - from reasoning_engine import ReasoningNode, ReasoningEdge, ReasoningPath
  - from causal_inference import CausalClaim, CausalGraph
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 20 more

### src/core/graph/temporal_evolution.py
- Module: src.core.graph.temporal_evolution
- Entry point: False
- Functions: 8
- Classes: 10
- Total imports: 31
- Internal imports: 5
- External imports: 26
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
  - from reasoning_engine import ReasoningNode, ReasoningEdge, ReasoningPath
  - from causal_inference import CausalClaim, CausalGraph
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 21 more

### src/core/ingestion/__init__.py
- Module: src.core.ingestion
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 3
- Internal imports: 3
- External imports: 0
- Internal imports:
  - from document_processor import DocumentProcessor
  - from file_loaders import get_file_loader, register_file_loader
  - from job_tracker import JobTracker, JobType, JobStatus, get_job_tracker

### src/core/ingestion/chunking_strategies.py
- Module: src.core.ingestion.chunking_strategies
- Entry point: False
- Functions: 23
- Classes: 14
- Total imports: 5
- Internal imports: 1
- External imports: 4
- Internal imports:
  - from utils.logger import get_logger
- External imports:
  - import abc
  - import re
  - import math
  - from typing import Any, Dict, List, Optional, Tuple

### src/core/ingestion/document_processor.py
- Module: src.core.ingestion.document_processor
- Entry point: False
- Functions: 2
- Classes: 1
- Total imports: 11
- Internal imports: 7
- External imports: 4
- Internal imports:
  - from memory.manager import MemoryManager, MemoryStoreRequest
  - from observability import get_tracer
  - from schemas.ingestion import ChunkMetadata, DocumentMetadata, IngestResponse, IngestionWarning
  - from utils.logger import get_logger
  - from chunking_strategies import chunk_content
  - from file_loaders import get_file_loader
  - from llm_context_enhancer import LLMContextEnhancer
- External imports:
  - import asyncio
  - import time
  - import uuid
  - from typing import Any, Dict, List, Optional

### src/core/ingestion/file_loaders.py
- Module: src.core.ingestion.file_loaders
- Entry point: False
- Functions: 18
- Classes: 8
- Total imports: 15
- Internal imports: 1
- External imports: 14
- Internal imports:
  - from utils.logger import get_logger
- External imports:
  - import abc
  - import asyncio
  - import base64
  - import csv
  - import io
  - ... and 9 more

### src/core/ingestion/file_watcher.py
- Module: src.core.ingestion.file_watcher
- Entry point: False
- Functions: 5
- Classes: 2
- Total imports: 15
- Internal imports: 4
- External imports: 11
- Internal imports:
  - from utils.simple_logger import get_logger
  - from utils.simple_config import get_settings
  - from document_processor import DocumentProcessor
  - from file_loaders import get_file_loader
- External imports:
  - import asyncio
  - import hashlib
  - import json
  - import os
  - import shutil
  - ... and 6 more

### src/core/ingestion/job_tracker.py
- Module: src.core.ingestion.job_tracker
- Entry point: False
- Functions: 13
- Classes: 4
- Total imports: 9
- Internal imports: 3
- External imports: 6
- Internal imports:
  - from cache.redis_cache import RedisCache
  - from utils.logger import get_logger
  - from cache.redis_cache import get_redis_cache
- External imports:
  - import asyncio
  - import time
  - import uuid
  - from datetime import datetime, timedelta
  - from enum import Enum
  - ... and 1 more

### src/core/ingestion/llm_context_enhancer.py
- Module: src.core.ingestion.llm_context_enhancer
- Entry point: False
- Functions: 4
- Classes: 2
- Total imports: 8
- Internal imports: 2
- External imports: 6
- Internal imports:
  - from utils.config import get_settings
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import json
  - import time
  - from typing import Any, Dict, List, Optional
  - import re
  - ... and 1 more

### src/core/interfaces/__init__.py
- Module: src.core.interfaces
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/core/interfaces/embeddings.py
- Module: src.core.interfaces.embeddings
- Entry point: False
- Functions: 3
- Classes: 5
- Total imports: 3
- Internal imports: 0
- External imports: 3
- External imports:
  - from abc import ABC, abstractmethod
  - from typing import Any, Dict, List, Optional
  - import numpy

### src/core/interfaces/graph_engine.py
- Module: src.core.interfaces.graph_engine
- Entry point: False
- Functions: 0
- Classes: 9
- Total imports: 5
- Internal imports: 0
- External imports: 5
- External imports:
  - from abc import ABC, abstractmethod
  - from dataclasses import dataclass
  - from datetime import datetime
  - from enum import Enum
  - from typing import Any, Dict, List, Optional, Union

### src/core/interfaces/hallucination_detector.py
- Module: src.core.interfaces.hallucination_detector
- Entry point: False
- Functions: 1
- Classes: 2
- Total imports: 3
- Internal imports: 0
- External imports: 3
- External imports:
  - from abc import ABC, abstractmethod
  - from typing import Dict, List, Any, Optional
  - from pydantic import BaseModel, Field

### src/core/interfaces/reranker.py
- Module: src.core.interfaces.reranker
- Entry point: False
- Functions: 3
- Classes: 8
- Total imports: 4
- Internal imports: 0
- External imports: 4
- External imports:
  - from abc import ABC, abstractmethod
  - from dataclasses import dataclass
  - from enum import Enum
  - from typing import Any, Dict, List, Optional, Tuple

### src/core/interfaces/vector_store.py
- Module: src.core.interfaces.vector_store
- Entry point: False
- Functions: 0
- Classes: 7
- Total imports: 5
- Internal imports: 0
- External imports: 5
- External imports:
  - from abc import ABC, abstractmethod
  - from dataclasses import dataclass
  - from datetime import datetime
  - from typing import Any, Dict, List, Optional, Tuple
  - import numpy

### src/core/learning/ab_testing.py
- Module: src.core.learning.ab_testing
- Entry point: False
- Functions: 15
- Classes: 10
- Total imports: 27
- Internal imports: 2
- External imports: 25
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import uuid
  - import hashlib
  - import numpy
  - import pandas
  - ... and 20 more

### src/core/learning/continuous_improvement.py
- Module: src.core.learning.continuous_improvement
- Entry point: False
- Functions: 27
- Classes: 9
- Total imports: 25
- Internal imports: 3
- External imports: 22
- Internal imports:
  - from online_learning import OnlineLearningSystem, OnlineLearningConfig, LearningTaskType, ModelType
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import os
  - import shutil
  - import pickle
  - import joblib
  - ... and 17 more

### src/core/learning/hyperparameter_optimization.py
- Module: src.core.learning.hyperparameter_optimization
- Entry point: False
- Functions: 20
- Classes: 9
- Total imports: 24
- Internal imports: 3
- External imports: 21
- Internal imports:
  - from online_learning import OnlineLearningConfig, LearningTaskType, ModelType
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import json
  - import pickle
  - import numpy
  - import pandas
  - ... and 16 more

### src/core/learning/online_learning.py
- Module: src.core.learning.online_learning
- Entry point: False
- Functions: 23
- Classes: 10
- Total imports: 25
- Internal imports: 2
- External imports: 23
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
  - from datetime import datetime, timedelta
  - ... and 18 more

### src/core/memory/__init__.py
- Module: src.core.memory
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/core/memory/manager.py
- Module: src.core.memory.manager
- Entry point: False
- Functions: 2
- Classes: 5
- Total imports: 16
- Internal imports: 7
- External imports: 9
- Internal imports:
  - from interfaces.embeddings import EmbeddingProvider
  - from interfaces.graph_engine import Entity, GraphEngine, Relationship
  - from interfaces.reranker import Reranker, RerankingCandidate, RerankingResult
  - from interfaces.vector_store import VectorDocument, VectorSearchResult, VectorStore
  - from utils.config import get_settings
  - from utils.logger import get_logger
  - from utils.registry import ProviderType, get_provider, get_provider_with_fallback
- External imports:
  - import asyncio
  - import time
  - import uuid
  - from dataclasses import dataclass
  - from datetime import datetime
  - ... and 4 more

### src/core/memory/models.py
- Module: src.core.memory.models
- Entry point: False
- Functions: 3
- Classes: 16
- Total imports: 5
- Internal imports: 0
- External imports: 5
- External imports:
  - from datetime import datetime
  - from enum import Enum
  - from typing import Any, Dict, List, Optional
  - from uuid import UUID, uuid4
  - from pydantic import BaseModel, Field, ConfigDict, field_validator

### src/core/memory/postgres_client.py
- Module: src.core.memory.postgres_client
- Entry point: False
- Functions: 1
- Classes: 2
- Total imports: 14
- Internal imports: 4
- External imports: 10
- Internal imports:
  - from utils.circuit_breaker import circuit_breaker
  - from utils.logger import get_logger
  - from interfaces.vector_store import VectorStore
  - from observability.tracing import trace_method
- External imports:
  - import asyncio
  - import json
  - from contextlib import asynccontextmanager
  - from datetime import datetime, timezone
  - from typing import Any, Dict, List, Optional, Tuple, AsyncIterator
  - ... and 5 more

### src/core/memory/retriever.py
- Module: src.core.memory.retriever
- Entry point: False
- Functions: 6
- Classes: 1
- Total imports: 14
- Internal imports: 9
- External imports: 5
- Internal imports:
  - from utils.logger import get_logger
  - from cache.redis_cache import CacheManager
  - from interfaces.embeddings import EmbeddingProvider
  - from interfaces.graph_engine import GraphEngine
  - from interfaces.reranker import Reranker
  - from observability.tracing import trace_method
  - from rag.hallucination_detector import HallucinationDetector
  - from models import MemorySearchResult, RetrievalContext, Memory, ConfidenceLevel
  - from postgres_client import PostgresClient
- External imports:
  - import asyncio
  - from typing import Any, Dict, List, Optional, Tuple
  - import numpy
  - import re
  - import re

### src/core/memory/structured_operations.py
- Module: src.core.memory.structured_operations
- Entry point: False
- Functions: 10
- Classes: 13
- Total imports: 18
- Internal imports: 4
- External imports: 14
- Internal imports:
  - from clients.vllm_client import VLLMClient
  - from embeddings.embedder import Embedder
  - from utils.config import settings
  - from manager import MemoryManager
- External imports:
  - import asyncio
  - import json
  - import hashlib
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any, Union, Tuple, Set
  - ... and 9 more

### src/core/observability/__init__.py
- Module: src.core.observability
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 3
- Internal imports: 3
- External imports: 0
- Internal imports:
  - from metrics import MemorySystemMetrics, MetricAggregation, MetricSnapshot, get_memory_metrics
  - from telemetry import TelemetryManager, get_telemetry, traced
  - from tracing import EnhancedTracer, TraceContext, get_tracer, traced_memory_operation, traced_rag_component

### src/core/observability/dashboard.py
- Module: src.core.observability.dashboard
- Entry point: False
- Functions: 15
- Classes: 10
- Total imports: 21
- Internal imports: 3
- External imports: 18
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import json
  - import time
  - import statistics
  - from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
  - ... and 13 more

### src/core/observability/metrics.py
- Module: src.core.observability.metrics
- Entry point: False
- Functions: 8
- Classes: 3
- Total imports: 9
- Internal imports: 2
- External imports: 7
- Internal imports:
  - from utils.logger import get_logger
  - from telemetry import get_telemetry
- External imports:
  - import asyncio
  - import time
  - from collections import defaultdict, deque
  - from dataclasses import dataclass, field
  - from datetime import datetime, timedelta
  - ... and 2 more

### src/core/observability/performance_optimized_telemetry.py
- Module: src.core.observability.performance_optimized_telemetry
- Entry point: False
- Functions: 28
- Classes: 4
- Total imports: 13
- Internal imports: 2
- External imports: 11
- Internal imports:
  - from utils.logger import get_logger
  - from utils.config import settings
- External imports:
  - import asyncio
  - import time
  - from collections import defaultdict, deque
  - from contextlib import asynccontextmanager, contextmanager
  - from typing import Any, Dict, Optional, Callable, List, Tuple, Union
  - ... and 6 more

### src/core/observability/telemetry.py
- Module: src.core.observability.telemetry
- Entry point: False
- Functions: 17
- Classes: 1
- Total imports: 25
- Internal imports: 4
- External imports: 21
- Internal imports:
  - from utils.config import get_settings
  - from utils.logger import get_logger
  - from performance_optimized_telemetry import get_telemetry
  - from telemetry_optimizer import get_telemetry_optimizer
- External imports:
  - import asyncio
  - import time
  - from contextlib import asynccontextmanager
  - from functools import wraps
  - from typing import Any, Callable, Dict, List, Optional, Union
  - ... and 16 more

### src/core/observability/telemetry_optimizer.py
- Module: src.core.observability.telemetry_optimizer
- Entry point: False
- Functions: 7
- Classes: 3
- Total imports: 9
- Internal imports: 3
- External imports: 6
- Internal imports:
  - from utils.logger import get_logger
  - from utils.config import settings
  - from performance_optimized_telemetry import get_telemetry, TelemetryLevel
- External imports:
  - import asyncio
  - import time
  - from typing import Dict, Any, Optional, List, Tuple
  - from dataclasses import dataclass
  - from enum import Enum
  - ... and 1 more

### src/core/observability/tracing.py
- Module: src.core.observability.tracing
- Entry point: False
- Functions: 17
- Classes: 2
- Total imports: 13
- Internal imports: 2
- External imports: 11
- Internal imports:
  - from utils.logger import get_logger
  - from telemetry import get_telemetry
- External imports:
  - import asyncio
  - import json
  - import time
  - from contextlib import asynccontextmanager, contextmanager
  - from dataclasses import dataclass
  - ... and 6 more

### src/core/optimization/query_optimizer.py
- Module: src.core.optimization.query_optimizer
- Entry point: False
- Functions: 27
- Classes: 8
- Total imports: 21
- Internal imports: 2
- External imports: 19
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import time
  - import re
  - import hashlib
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union
  - ... and 14 more

### src/core/prediction/__init__.py
- Module: src.core.prediction
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 4
- Internal imports: 4
- External imports: 0
- Internal imports:
  - from usage_analyzer import UsageAnalyzer, UsagePattern, AccessPattern, UserBehaviorModel, PopularityScore, PatternType, AccessFrequency
  - from auto_archiver import AutoArchiver, ArchivingPolicy, ImportanceScore, ArchivingRule, ArchivingAction, PolicyEngine, RestoreTrigger
  - from preloader import PredictivePreloader, CachePrediction, QueryAnticipator, MarkovChainPredictor, PreloadingStrategy, PriorityQueue, SuccessTracker
  - from lifecycle import MemoryLifecycleOptimizer, LifecycleStage, TransitionPrediction, OptimizationRecommendation, PerformanceMetrics, AgingAlgorithm

### src/core/prediction/auto_archiver.py
- Module: src.core.prediction.auto_archiver
- Entry point: False
- Functions: 14
- Classes: 10
- Total imports: 22
- Internal imports: 4
- External imports: 18
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from usage_analyzer import UsageAnalyzer, UsagePattern
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 13 more

### src/core/prediction/lifecycle.py
- Module: src.core.prediction.lifecycle
- Entry point: False
- Functions: 14
- Classes: 10
- Total imports: 24
- Internal imports: 5
- External imports: 19
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from usage_analyzer import UsageAnalyzer, UsagePattern
  - from auto_archiver import AutoArchiver, ImportanceScore
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
  - from datetime import datetime, timedelta
  - ... and 14 more

### src/core/prediction/preloader.py
- Module: src.core.prediction.preloader
- Entry point: False
- Functions: 28
- Classes: 11
- Total imports: 22
- Internal imports: 4
- External imports: 18
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from usage_analyzer import UsageAnalyzer, UsagePattern
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable, Deque
  - from datetime import datetime, timedelta
  - ... and 13 more

### src/core/prediction/usage_analyzer.py
- Module: src.core.prediction.usage_analyzer
- Entry point: False
- Functions: 5
- Classes: 7
- Total imports: 25
- Internal imports: 3
- External imports: 22
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Tuple, Union
  - from datetime import datetime, timedelta
  - ... and 17 more

### src/core/providers/__init__.py
- Module: src.core.providers
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/core/rag/__init__.py
- Module: src.core.rag
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 6
- Internal imports: 6
- External imports: 0
- Internal imports:
  - from multimodal import MultiModalProcessor, MultiModalContent, MultiModalSearchResult, ModalityType, DocumentType, CodeLanguage
  - from chunk_linking import ContextualChunkLinker, TextChunk, ContextualLink, ChunkLinkingResult, LinkType, ChunkType, LinkStrength
  - from dynamic_reranking import DynamicReranker, QueryIntent, QueryContext, RerankingStrategy, ContextType, RankingFeatures, RerankingResult
  - from intent_detector import IntentAwareRetriever, RetrievalStrategy, RetrievalMode, FilterCriteria, RetrievalContext, RetrievalResult, RetrievalAnalytics
  - from reranker import Reranker, RerankingResult
  - from hallucination_detector import HallucinationDetector, HallucinationResult, HallucinationType, ConfidenceLevel, ValidationMetrics

### src/core/rag/chunk_linking.py
- Module: src.core.rag.chunk_linking
- Entry point: False
- Functions: 3
- Classes: 7
- Total imports: 26
- Internal imports: 4
- External imports: 22
- Internal imports:
  - from models.memory import Memory
  - from embeddings.embedder import Embedder
  - from cache.redis_cache import RedisCache
  - from utils.config import settings
- External imports:
  - import asyncio
  - from typing import List, Dict, Optional, Tuple, Set, Union, Any
  - from datetime import datetime
  - from dataclasses import dataclass, field
  - from enum import Enum
  - ... and 17 more

### src/core/rag/dynamic_reranking.py
- Module: src.core.rag.dynamic_reranking
- Entry point: False
- Functions: 8
- Classes: 7
- Total imports: 25
- Internal imports: 4
- External imports: 21
- Internal imports:
  - from models.memory import Memory
  - from embeddings.embedder import Embedder
  - from cache.redis_cache import RedisCache
  - from utils.config import settings
- External imports:
  - import asyncio
  - from typing import List, Dict, Optional, Tuple, Union, Any, Callable
  - from datetime import datetime
  - from dataclasses import dataclass, field
  - from enum import Enum
  - ... and 16 more

### src/core/rag/hallucination_detector.py
- Module: src.core.rag.hallucination_detector
- Entry point: False
- Functions: 12
- Classes: 4
- Total imports: 11
- Internal imports: 3
- External imports: 8
- Internal imports:
  - from interfaces.embeddings import EmbeddingProvider
  - from utils.config import get_settings
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import time
  - from dataclasses import dataclass
  - from enum import Enum
  - from typing import Any, Dict, List, Optional, Tuple
  - ... and 3 more

### src/core/rag/intent_detector.py
- Module: src.core.rag.intent_detector
- Entry point: False
- Functions: 5
- Classes: 7
- Total imports: 26
- Internal imports: 6
- External imports: 20
- Internal imports:
  - from models.memory import Memory
  - from embeddings.embedder import Embedder
  - from cache.redis_cache import RedisCache
  - from dynamic_reranking import QueryIntent, QueryContext, DynamicReranker
  - from chunk_linking import ContextualChunkLinker, TextChunk, LinkType
  - from utils.config import settings
- External imports:
  - import asyncio
  - from typing import List, Dict, Optional, Tuple, Union, Any, Set
  - from datetime import datetime, timedelta
  - from dataclasses import dataclass, field
  - from enum import Enum
  - ... and 15 more

### src/core/rag/multimodal.py
- Module: src.core.rag.multimodal
- Entry point: False
- Functions: 8
- Classes: 6
- Total imports: 33
- Internal imports: 4
- External imports: 29
- Internal imports:
  - from models.memory import Memory
  - from embeddings.embedder import Embedder
  - from cache.redis_cache import RedisCache
  - from utils.config import settings
- External imports:
  - import asyncio
  - from typing import List, Dict, Optional, Tuple, Union, Any, BinaryIO
  - from datetime import datetime
  - from dataclasses import dataclass, field
  - from enum import Enum
  - ... and 24 more

### src/core/rag/pipeline_manager.py
- Module: src.core.rag.pipeline_manager
- Entry point: False
- Functions: 2
- Classes: 6
- Total imports: 18
- Internal imports: 9
- External imports: 9
- Internal imports:
  - from multimodal import MultiModalProcessor, MultiModalContent, ModalityType
  - from chunk_linking import ContextualChunkLinker, TextChunk, ChunkLinkingResult
  - from dynamic_reranking import DynamicReranker, QueryIntent, QueryContext
  - from intent_detector import IntentAwareRetriever, RetrievalContext, RetrievalResult
  - from hallucination_detector import HallucinationDetector, HallucinationResult
  - from embeddings.embedder import Embedder
  - from memory.manager import MemoryManager
  - from graph.neo4j_client import Neo4jClient
  - from cache.redis_cache import RedisCache
- External imports:
  - import asyncio
  - import time
  - from typing import Dict, List, Optional, Any, Union, Tuple
  - from dataclasses import dataclass, field
  - from enum import Enum
  - ... and 4 more

### src/core/rag/reranker.py
- Module: src.core.rag.reranker
- Entry point: False
- Functions: 6
- Classes: 2
- Total imports: 11
- Internal imports: 6
- External imports: 5
- Internal imports:
  - from utils.logger import get_logger
  - from utils.circuit_breaker import circuit_breaker
  - from interfaces.reranker import Reranker
  - from observability.tracing import trace_method
  - from cache.redis_cache import CacheManager
  - from vllm_reranker import VLLMReranker
- External imports:
  - import asyncio
  - from typing import List, Optional, Tuple, Dict, Any
  - import numpy
  - import torch
  - from sentence_transformers import CrossEncoder

### src/core/rag/retrieval.py
- Module: src.core.rag.retrieval
- Entry point: False
- Functions: 8
- Classes: 4
- Total imports: 13
- Internal imports: 6
- External imports: 7
- Internal imports:
  - from interfaces.embeddings import EmbeddingProvider
  - from interfaces.graph_engine import Entity, GraphEngine, Relationship
  - from interfaces.reranker import Reranker, RerankingCandidate, RerankingResult
  - from interfaces.vector_store import VectorSearchResult, VectorStore
  - from utils.logger import get_logger
  - from utils.registry import ProviderType, get_provider
- External imports:
  - import asyncio
  - import logging
  - from dataclasses import dataclass
  - from datetime import datetime
  - from enum import Enum
  - ... and 2 more

### src/core/rag/scorer.py
- Module: src.core.rag.scorer
- Entry point: False
- Functions: 9
- Classes: 1
- Total imports: 6
- Internal imports: 3
- External imports: 3
- Internal imports:
  - from utils.logger import get_logger
  - from memory.models import ConfidenceLevel, MemorySearchResult
  - from observability.tracing import trace_method
- External imports:
  - from datetime import datetime
  - from typing import Dict, List, Optional, Tuple
  - import numpy

### src/core/rag/vllm_reranker.py
- Module: src.core.rag.vllm_reranker
- Entry point: False
- Functions: 5
- Classes: 1
- Total imports: 12
- Internal imports: 5
- External imports: 7
- Internal imports:
  - from utils.logger import get_logger
  - from utils.circuit_breaker import circuit_breaker
  - from interfaces.reranker import Reranker
  - from observability.tracing import trace_method
  - from cache.cache_manager import CacheManager
- External imports:
  - import asyncio
  - import json
  - import time
  - from typing import Any, Dict, List, Optional, Tuple
  - import aiohttp
  - ... and 2 more

### src/core/schemas/ingestion.py
- Module: src.core.schemas.ingestion
- Entry point: False
- Functions: 2
- Classes: 10
- Total imports: 4
- Internal imports: 0
- External imports: 4
- External imports:
  - import uuid
  - from datetime import datetime
  - from typing import Any, Dict, List, Literal, Optional, Union
  - from pydantic import BaseModel, Field, HttpUrl, validator

### src/core/security/audit.py
- Module: src.core.security.audit
- Entry point: False
- Functions: 19
- Classes: 9
- Total imports: 22
- Internal imports: 2
- External imports: 20
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import hashlib
  - import hmac
  - import json
  - import time
  - ... and 15 more

### src/core/security/encryption.py
- Module: src.core.security.encryption
- Entry point: False
- Functions: 20
- Classes: 12
- Total imports: 29
- Internal imports: 3
- External imports: 26
- Internal imports:
  - from models.memory import Memory
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import secrets
  - import hashlib
  - import hmac
  - import json
  - ... and 21 more

### src/core/security/rbac.py
- Module: src.core.security.rbac
- Entry point: False
- Functions: 34
- Classes: 16
- Total imports: 25
- Internal imports: 2
- External imports: 23
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import hashlib
  - import secrets
  - import hmac
  - import json
  - ... and 18 more

### src/core/services/__init__.py
- Module: src.core.services
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from file_watcher_service import FileWatcherServiceManager, get_file_watcher_manager

### src/core/services/file_watcher_service.py
- Module: src.core.services.file_watcher_service
- Entry point: False
- Functions: 3
- Classes: 1
- Total imports: 4
- Internal imports: 2
- External imports: 2
- Internal imports:
  - from ingestion.file_watcher import FileWatcherService, get_file_watcher_service
  - from utils.simple_logger import get_logger
- External imports:
  - import asyncio
  - from typing import Optional

### src/core/synthesis/__init__.py
- Module: src.core.synthesis
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 4
- Internal imports: 4
- External imports: 0
- Internal imports:
  - from deduplication import DeduplicationEngine, DuplicateGroup, DuplicateType, MergeStrategy, DuplicationMetrics
  - from summarization import SummarizationEngine, MemorySummary, SummarizationType, SummaryQuality, QualityMetrics
  - from pattern_detector import PatternDetector, PatternCluster, PatternType, ClusteringMethod, PatternInsight, KnowledgeGap, PatternAnalysisResult
  - from temporal_analysis import TemporalAnalyzer, ConceptEvolution, EvolutionType, TrendDirection, TemporalInsight, LearningProgression, TemporalAnalysisResult

### src/core/synthesis/deduplication.py
- Module: src.core.synthesis.deduplication
- Entry point: False
- Functions: 3
- Classes: 5
- Total imports: 19
- Internal imports: 4
- External imports: 15
- Internal imports:
  - from models.memory import Memory, MemoryUpdate
  - from embeddings.embedder import Embedder
  - from cache.redis_cache import RedisCache
  - from utils.config import settings
- External imports:
  - import hashlib
  - import asyncio
  - from typing import List, Dict, Tuple, Optional, Set, Any
  - from datetime import datetime
  - from dataclasses import dataclass, field
  - ... and 10 more

### src/core/synthesis/pattern_detector.py
- Module: src.core.synthesis.pattern_detector
- Entry point: False
- Functions: 4
- Classes: 7
- Total imports: 27
- Internal imports: 4
- External imports: 23
- Internal imports:
  - from models.memory import Memory
  - from embeddings.embedder import Embedder
  - from cache.redis_cache import RedisCache
  - from utils.config import settings
- External imports:
  - import asyncio
  - from typing import List, Dict, Optional, Tuple, Set, Any, Union
  - from datetime import datetime, timedelta
  - from dataclasses import dataclass, field
  - from enum import Enum
  - ... and 18 more

### src/core/synthesis/summarization.py
- Module: src.core.synthesis.summarization
- Entry point: False
- Functions: 8
- Classes: 5
- Total imports: 26
- Internal imports: 6
- External imports: 20
- Internal imports:
  - from models.memory import Memory
  - from embeddings.embedder import Embedder
  - from rag.hallucination import HallucinationDetector
  - from cache.redis_cache import RedisCache
  - from utils.config import settings
  - from clients.vllm_client import VLLMClient
- External imports:
  - import asyncio
  - from typing import List, Dict, Optional, Tuple, Set, Any, Union
  - from datetime import datetime
  - from dataclasses import dataclass
  - from enum import Enum
  - ... and 15 more

### src/core/synthesis/temporal_analysis.py
- Module: src.core.synthesis.temporal_analysis
- Entry point: False
- Functions: 4
- Classes: 8
- Total imports: 24
- Internal imports: 4
- External imports: 20
- Internal imports:
  - from models.memory import Memory
  - from embeddings.embedder import Embedder
  - from cache.redis_cache import RedisCache
  - from utils.config import settings
- External imports:
  - import asyncio
  - from typing import List, Dict, Optional, Tuple, Set, Any, Union
  - from datetime import datetime, timedelta
  - from dataclasses import dataclass, field
  - from enum import Enum
  - ... and 15 more

### src/core/utils/__init__.py
- Module: src.core.utils
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/core/utils/circuit_breaker.py
- Module: src.core.utils.circuit_breaker
- Entry point: False
- Functions: 6
- Classes: 8
- Total imports: 7
- Internal imports: 1
- External imports: 6
- Internal imports:
  - from logger import get_logger
- External imports:
  - import asyncio
  - import time
  - from contextlib import asynccontextmanager
  - from dataclasses import dataclass, field
  - from enum import Enum
  - ... and 1 more

### src/core/utils/config.py
- Module: src.core.utils.config
- Entry point: False
- Functions: 19
- Classes: 14
- Total imports: 9
- Internal imports: 0
- External imports: 9
- External imports:
  - import os
  - import re
  - from dataclasses import dataclass
  - from pathlib import Path
  - from typing import Any, Dict, Optional, Union
  - ... and 4 more

### src/core/utils/database.py
- Module: src.core.utils.database
- Entry point: False
- Functions: 18
- Classes: 8
- Total imports: 19
- Internal imports: 2
- External imports: 17
- Internal imports:
  - from circuit_breaker import AsyncCircuitBreaker, CircuitBreakerConfig, get_circuit_breaker
  - from logger import get_logger
- External imports:
  - import asyncio
  - import time
  - import psutil
  - import numpy
  - from abc import ABC, abstractmethod
  - ... and 12 more

### src/core/utils/logger.py
- Module: src.core.utils.logger
- Entry point: False
- Functions: 23
- Classes: 3
- Total imports: 13
- Internal imports: 1
- External imports: 12
- Internal imports:
  - from config import get_settings
- External imports:
  - import logging
  - import logging.handlers
  - import sys
  - from contextvars import ContextVar
  - from datetime import datetime
  - ... and 7 more

### src/core/utils/registry.py
- Module: src.core.utils.registry
- Entry point: False
- Functions: 3
- Classes: 3
- Total imports: 9
- Internal imports: 2
- External imports: 7
- Internal imports:
  - from config import get_provider_config
  - from simple_logger import get_logger
- External imports:
  - import asyncio
  - import importlib
  - import inspect
  - from abc import ABC, abstractmethod
  - from dataclasses import dataclass, field
  - ... and 2 more

### src/core/utils/simple_config.py
- Module: src.core.utils.simple_config
- Entry point: False
- Functions: 19
- Classes: 2
- Total imports: 5
- Internal imports: 0
- External imports: 5
- External imports:
  - import os
  - import re
  - from pathlib import Path
  - from typing import Any, Dict, Optional, Union
  - import yaml

### src/core/utils/simple_logger.py
- Module: src.core.utils.simple_logger
- Entry point: False
- Functions: 11
- Classes: 2
- Total imports: 5
- Internal imports: 0
- External imports: 5
- External imports:
  - import json
  - import logging
  - import sys
  - from datetime import datetime
  - from typing import Any, Dict, Optional

### src/dashboard/analytics/__init__.py
- Module: src.dashboard.analytics
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/dashboard/analytics/local_roi.py
- Module: src.dashboard.analytics.local_roi
- Entry point: False
- Functions: 20
- Classes: 10
- Total imports: 32
- Internal imports: 4
- External imports: 28
- Internal imports:
  - from core.memory.manager import MemoryManager
  - from core.graph.neo4j_client import Neo4jClient
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import json
  - import math
  - import hashlib
  - from datetime import datetime, timedelta, timezone
  - ... and 23 more

### src/dashboard/analytics/local_usage.py
- Module: src.dashboard.analytics.local_usage
- Entry point: False
- Functions: 15
- Classes: 9
- Total imports: 34
- Internal imports: 4
- External imports: 30
- Internal imports:
  - from core.memory.manager import MemoryManager
  - from core.graph.neo4j_client import Neo4jClient
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import json
  - import math
  - import hashlib
  - from datetime import datetime, timedelta, timezone
  - ... and 25 more

### src/dashboard/insights/__init__.py
- Module: src.dashboard.insights
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/dashboard/insights/local_gaps.py
- Module: src.dashboard.insights.local_gaps
- Entry point: False
- Functions: 15
- Classes: 7
- Total imports: 34
- Internal imports: 5
- External imports: 29
- Internal imports:
  - from core.memory.manager import MemoryManager
  - from core.graph.neo4j_client import Neo4jClient
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
  - from suggestions.gaps.local_detector import LocalKnowledgeGapDetector, KnowledgeGap, GapType
- External imports:
  - import asyncio
  - import json
  - import math
  - import hashlib
  - from datetime import datetime, timedelta, timezone
  - ... and 24 more

### src/dashboard/visualization/__init__.py
- Module: src.dashboard.visualization
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/dashboard/visualization/local_network.py
- Module: src.dashboard.visualization.local_network
- Entry point: False
- Functions: 10
- Classes: 9
- Total imports: 27
- Internal imports: 4
- External imports: 23
- Internal imports:
  - from core.memory.manager import MemoryManager
  - from core.graph.neo4j_client import Neo4jClient
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import json
  - import math
  - import hashlib
  - from datetime import datetime, timedelta
  - ... and 18 more

### src/infrastructure/scaling/auto_scaler.py
- Module: src.infrastructure.scaling.auto_scaler
- Entry point: False
- Functions: 21
- Classes: 10
- Total imports: 16
- Internal imports: 2
- External imports: 14
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import psutil
  - import time
  - import numpy
  - from typing import Dict, List, Optional, Set, Any, Tuple, Callable
  - ... and 9 more

### src/migrations/sql/__init__.py
- Module: src.migrations.sql
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/suggestions/connections/__init__.py
- Module: src.suggestions.connections
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/suggestions/connections/local_connector.py
- Module: src.suggestions.connections.local_connector
- Entry point: False
- Functions: 16
- Classes: 7
- Total imports: 25
- Internal imports: 6
- External imports: 19
- Internal imports:
  - from core.embeddings.embedder import Embedder
  - from core.memory.manager import MemoryManager
  - from core.graph.neo4j_client import Neo4jClient
  - from core.utils.config import settings
  - from core.embeddings.embedder import Embedder
  - from core.memory.manager import MemoryManager
- External imports:
  - import asyncio
  - import math
  - import json
  - import hashlib
  - from datetime import datetime, timedelta
  - ... and 14 more

### src/suggestions/gaps/__init__.py
- Module: src.suggestions.gaps
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/suggestions/gaps/local_detector.py
- Module: src.suggestions.gaps.local_detector
- Entry point: False
- Functions: 45
- Classes: 8
- Total imports: 26
- Internal imports: 6
- External imports: 20
- Internal imports:
  - from core.embeddings.embedder import Embedder
  - from core.memory.manager import MemoryManager
  - from core.graph.neo4j_client import Neo4jClient
  - from core.utils.config import settings
  - from core.embeddings.embedder import Embedder
  - from core.memory.manager import MemoryManager
- External imports:
  - import asyncio
  - import math
  - import json
  - import hashlib
  - from datetime import datetime, timedelta
  - ... and 15 more

### src/suggestions/organization/__init__.py
- Module: src.suggestions.organization
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/suggestions/organization/local_recommender.py
- Module: src.suggestions.organization.local_recommender
- Entry point: False
- Functions: 27
- Classes: 7
- Total imports: 25
- Internal imports: 6
- External imports: 19
- Internal imports:
  - from core.embeddings.embedder import Embedder
  - from core.memory.manager import MemoryManager
  - from core.graph.neo4j_client import Neo4jClient
  - from core.utils.config import settings
  - from core.embeddings.embedder import Embedder
  - from core.memory.manager import MemoryManager
- External imports:
  - import asyncio
  - import math
  - import json
  - import hashlib
  - from datetime import datetime, timedelta
  - ... and 14 more

### src/suggestions/related/__init__.py
- Module: src.suggestions.related
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 0
- Internal imports: 0
- External imports: 0

### src/suggestions/related/local_suggester.py
- Module: src.suggestions.related.local_suggester
- Entry point: False
- Functions: 12
- Classes: 6
- Total imports: 25
- Internal imports: 5
- External imports: 20
- Internal imports:
  - from core.embeddings.embedder import Embedder
  - from core.memory.manager import MemoryManager
  - from core.utils.config import settings
  - from core.embeddings.embedder import Embedder
  - from core.memory.manager import MemoryManager
- External imports:
  - import asyncio
  - import math
  - import pickle
  - from datetime import datetime, timedelta
  - from typing import Dict, List, Optional, Any, Tuple, Set, Union
  - ... and 15 more

### tests/integration/api/test_memory_endpoints.py
- Module: tests.integration.api.test_memory_endpoints
- Entry point: False
- Functions: 3
- Classes: 1
- Total imports: 7
- Internal imports: 2
- External imports: 5
- Internal imports:
  - from src.api.app import get_app
  - from src.core.memory.manager import MemoryManager
- External imports:
  - import asyncio
  - from unittest.mock import AsyncMock, patch
  - import pytest
  - from fastapi.testclient import TestClient
  - from httpx import AsyncClient

### tests/unit/core/test_embeddings.py
- Module: tests.unit.core.test_embeddings
- Entry point: False
- Functions: 4
- Classes: 1
- Total imports: 7
- Internal imports: 2
- External imports: 5
- Internal imports:
  - from src.core.interfaces.embeddings import EmbeddingError, EmbeddingProvider
  - from src.core.providers.embeddings.huggingface import HuggingFaceEmbedder
- External imports:
  - import asyncio
  - from typing import List
  - from unittest.mock import AsyncMock, Mock, patch
  - import numpy
  - import pytest

### tests/unit/core/test_hallucination_detector.py
- Module: tests.unit.core.test_hallucination_detector
- Entry point: False
- Functions: 4
- Classes: 1
- Total imports: 5
- Internal imports: 2
- External imports: 3
- Internal imports:
  - from src.core.interfaces.embeddings import EmbeddingProvider
  - from src.core.rag.hallucination_detector import HallucinationDetector
- External imports:
  - from typing import Any, Dict, List
  - from unittest.mock import AsyncMock, Mock, patch
  - import pytest

### tests/unit/core/test_memory_manager.py
- Module: tests.unit.core.test_memory_manager
- Entry point: False
- Functions: 4
- Classes: 1
- Total imports: 11
- Internal imports: 6
- External imports: 5
- Internal imports:
  - from src.core.interfaces.embeddings import EmbeddingProvider
  - from src.core.interfaces.graph_engine import GraphEngine
  - from src.core.interfaces.reranker import Reranker
  - from src.core.interfaces.vector_store import Memory, VectorStore
  - from src.core.memory.manager import MemoryManager
  - from src.core.rag.hallucination_detector import HallucinationDetector
- External imports:
  - import asyncio
  - from datetime import datetime
  - from typing import Any, Dict, List
  - from unittest.mock import AsyncMock, Mock, patch
  - import pytest

### src/core/personalization/confidence/adaptive_confidence.py
- Module: src.core.personalization.confidence.adaptive_confidence
- Entry point: False
- Functions: 22
- Classes: 9
- Total imports: 26
- Internal imports: 2
- External imports: 24
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
  - from datetime import datetime, timedelta
  - ... and 19 more

### src/core/personalization/preferences/user_preference_engine.py
- Module: src.core.personalization.preferences.user_preference_engine
- Entry point: False
- Functions: 26
- Classes: 10
- Total imports: 31
- Internal imports: 2
- External imports: 29
- Internal imports:
  - from core.cache.redis_cache import RedisCache
  - from core.utils.config import settings
- External imports:
  - import asyncio
  - import numpy
  - import pandas
  - from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
  - from datetime import datetime, timedelta
  - ... and 24 more

### src/core/providers/embeddings/__init__.py
- Module: src.core.providers.embeddings
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from huggingface import HuggingFaceEmbeddingProvider

### src/core/providers/embeddings/embedder.py
- Module: src.core.providers.embeddings.embedder
- Entry point: False
- Functions: 7
- Classes: 1
- Total imports: 9
- Internal imports: 5
- External imports: 4
- Internal imports:
  - from utils.logger import get_logger
  - from utils.config import settings
  - from interfaces.embeddings import EmbeddingProvider
  - from observability.tracing import trace_method
  - from registry import get_embedding_provider
- External imports:
  - import asyncio
  - from typing import List, Optional, Dict, Any
  - import numpy
  - import torch

### src/core/providers/embeddings/huggingface.py
- Module: src.core.providers.embeddings.huggingface
- Entry point: False
- Functions: 9
- Classes: 1
- Total imports: 10
- Internal imports: 2
- External imports: 8
- Internal imports:
  - from interfaces.embeddings import EmbeddingGenerationError, EmbeddingInitializationError, EmbeddingProvider, EmbeddingProviderError
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import os
  - import time
  - from typing import Any, Dict, List, Optional, Union
  - import numpy
  - ... and 3 more

### src/core/providers/embeddings/registry.py
- Module: src.core.providers.embeddings.registry
- Entry point: False
- Functions: 9
- Classes: 1
- Total imports: 5
- Internal imports: 2
- External imports: 3
- Internal imports:
  - from interfaces.embeddings import EmbeddingProvider
  - from huggingface import HuggingFaceEmbeddingProvider
- External imports:
  - import logging
  - from abc import ABC
  - from typing import Any, Dict, List, Optional, Type

### src/core/providers/graph_engines/__init__.py
- Module: src.core.providers.graph_engines
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 2
- Internal imports: 2
- External imports: 0
- Internal imports:
  - from neo4j import Neo4jEngine
  - from registry import GraphEngineProviderRegistry, create_graph_engine, get_available_providers

### src/core/providers/graph_engines/neo4j.py
- Module: src.core.providers.graph_engines.neo4j
- Entry point: False
- Functions: 1
- Classes: 1
- Total imports: 13
- Internal imports: 4
- External imports: 9
- Internal imports:
  - from interfaces.graph_engine import Entity, GraphEngine, GraphEngineError, GraphEngineInitializationError, GraphEngineOperationError, GraphSearchResult
  - from interfaces.graph_engine import Relationship
  - from interfaces.graph_engine import RelationshipType
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import json
  - import time
  - import uuid
  - from dataclasses import asdict
  - ... and 4 more

### src/core/providers/graph_engines/registry.py
- Module: src.core.providers.graph_engines.registry
- Entry point: False
- Functions: 9
- Classes: 1
- Total imports: 5
- Internal imports: 2
- External imports: 3
- Internal imports:
  - from interfaces.graph_engine import GraphEngine
  - from neo4j import Neo4jEngine
- External imports:
  - import logging
  - from abc import ABC
  - from typing import Any, Dict, List, Optional, Type

### src/core/providers/rerankers/__init__.py
- Module: src.core.providers.rerankers
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 2
- Internal imports: 2
- External imports: 0
- Internal imports:
  - from cross_encoder import CrossEncoderReranker
  - from registry import RerankerProviderRegistry, create_reranker, get_available_providers

### src/core/providers/rerankers/cross_encoder.py
- Module: src.core.providers.rerankers.cross_encoder
- Entry point: False
- Functions: 10
- Classes: 1
- Total imports: 10
- Internal imports: 2
- External imports: 8
- Internal imports:
  - from interfaces.reranker import Reranker, RerankerError, RerankerInitializationError, RerankerOperationError, RerankerType, RerankingCandidate, RerankingResult
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import os
  - import time
  - from typing import Any, Dict, List, Optional, Tuple
  - import numpy
  - ... and 3 more

### src/core/providers/rerankers/registry.py
- Module: src.core.providers.rerankers.registry
- Entry point: False
- Functions: 9
- Classes: 1
- Total imports: 6
- Internal imports: 3
- External imports: 3
- Internal imports:
  - from interfaces.reranker import Reranker
  - from cross_encoder import CrossEncoderReranker
  - from rag.vllm_reranker import VLLMReranker
- External imports:
  - import logging
  - from abc import ABC
  - from typing import Any, Dict, List, Optional, Type

### src/core/providers/vector_stores/__init__.py
- Module: src.core.providers.vector_stores
- Entry point: False
- Functions: 0
- Classes: 0
- Total imports: 1
- Internal imports: 1
- External imports: 0
- Internal imports:
  - from pgvector import PgVectorStore

### src/core/providers/vector_stores/pgvector.py
- Module: src.core.providers.vector_stores.pgvector
- Entry point: False
- Functions: 4
- Classes: 1
- Total imports: 11
- Internal imports: 3
- External imports: 8
- Internal imports:
  - from interfaces.vector_store import VectorDocument, VectorSearchResult, VectorStore, VectorStoreError, VectorStoreInitializationError, VectorStoreOperationError
  - from utils.database import PostgreSQLManager
  - from utils.logger import get_logger
- External imports:
  - import asyncio
  - import json
  - import time
  - from dataclasses import asdict
  - from datetime import datetime
  - ... and 3 more

### src/core/providers/vector_stores/registry.py
- Module: src.core.providers.vector_stores.registry
- Entry point: False
- Functions: 9
- Classes: 1
- Total imports: 5
- Internal imports: 2
- External imports: 3
- Internal imports:
  - from interfaces.vector_store import VectorStore
  - from pgvector import PgVectorStore
- External imports:
  - import logging
  - from abc import ABC
  - from typing import Any, Dict, List, Optional, Type
