#!/usr/bin/env python3
"""
Test script for database setup verification.
This script tests the database setup created by setup_existing_databases.sh
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_postgresql_setup():
    """Test PostgreSQL setup and schema."""
    print("🧪 Testing PostgreSQL setup...")
    
    try:
        import asyncpg
        from core.utils.simple_config import get_setting
        
        # Get connection details from config or environment
        host = get_setting("databases.postgresql.host", os.getenv("POSTGRES_HOST", "localhost"))
        port = get_setting("databases.postgresql.port", int(os.getenv("POSTGRES_PORT", 5432)))
        database = get_setting("databases.postgresql.database", os.getenv("POSTGRES_DB", "tyra_memory"))
        user = get_setting("databases.postgresql.user", os.getenv("POSTGRES_USER", "tyra"))
        password = get_setting("databases.postgresql.password", os.getenv("POSTGRES_PASSWORD", ""))
        
        if not password:
            print("❌ PostgreSQL password not found in config or environment")
            return False
            
        # Test connection
        conn = await asyncpg.connect(
            host=host, port=port, database=database, 
            user=user, password=password
        )
        
        # Test basic functionality
        result = await conn.fetchval("SELECT version()")
        print(f"✅ PostgreSQL connected: {result.split()[0:2]}")
        
        # Test pgvector extension
        try:
            distance = await conn.fetchval("SELECT '[1,2,3]'::vector(3) <-> '[4,5,6]'::vector(3)")
            print(f"✅ pgvector working: distance = {distance:.3f}")
        except Exception as e:
            print(f"❌ pgvector test failed: {e}")
            return False
            
        # Check schema tables
        table_count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        print(f"✅ Schema tables: {table_count} tables found")
        
        # Test key tables exist
        key_tables = ['memories', 'memory_chunks', 'entities', 'relationships']
        for table in key_tables:
            exists = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1", 
                table
            )
            if exists:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' missing")
                return False
                
        await conn.close()
        return True
        
    except ImportError:
        print("❌ asyncpg not available - install with: pip install asyncpg")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL test failed: {e}")
        return False

async def test_redis_setup():
    """Test Redis setup."""
    print("\n🧪 Testing Redis setup...")
    
    try:
        import redis.asyncio as redis
        from core.utils.simple_config import get_setting
        
        # Get connection details
        host = get_setting("cache.redis.host", os.getenv("REDIS_HOST", "localhost"))
        port = get_setting("cache.redis.port", int(os.getenv("REDIS_PORT", 6379)))
        password = get_setting("cache.redis.password", os.getenv("REDIS_PASSWORD", None))
        
        # Test connection
        r = redis.Redis(host=host, port=port, password=password, decode_responses=True)
        
        # Test ping
        pong = await r.ping()
        if pong:
            print("✅ Redis connected and responding to ping")
        else:
            print("❌ Redis ping failed")
            return False
            
        # Test basic operations
        await r.set("test_key", "test_value", ex=60)
        value = await r.get("test_key")
        if value == "test_value":
            print("✅ Redis set/get operations working")
        else:
            print("❌ Redis set/get operations failed")
            return False
            
        await r.delete("test_key")
        await r.aclose()
        return True
        
    except ImportError:
        print("❌ redis not available - install with: pip install redis")
        return False
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

async def test_neo4j_setup():
    """Test Neo4j setup."""
    print("\n🧪 Testing Neo4j setup...")
    
    try:
        from neo4j import GraphDatabase
        from core.utils.simple_config import get_setting
        
        # Get connection details
        host = get_setting("graph.neo4j.host", os.getenv("NEO4J_HOST", "localhost"))
        port = get_setting("graph.neo4j.port", int(os.getenv("NEO4J_PORT", 7687)))
        user = get_setting("graph.neo4j.user", os.getenv("NEO4J_USER", "neo4j"))
        password = get_setting("graph.neo4j.password", os.getenv("NEO4J_PASSWORD", "neo4j123"))
        
        # Test connection
        driver = GraphDatabase.driver(f"bolt://{host}:{port}", auth=(user, password))
        
        with driver.session() as session:
            # Test basic query
            result = session.run("RETURN 'Neo4j connected' as status")
            status_result = result.single()
            if status_result:
                print(f"✅ Neo4j connected: {status_result['status']}")
            else:
                print("❌ Neo4j query failed")
                return False
                
            # Test node count
            count_result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = count_result.single()['node_count']
            print(f"✅ Neo4j nodes: {node_count} nodes in graph")
        
        driver.close()
        return True
        
    except ImportError:
        print("❌ neo4j driver not available - install with: pip install neo4j")
        return False
    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")
        return False

async def test_config_files():
    """Test configuration files."""
    print("\n🧪 Testing configuration files...")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check key variables
        env_content = env_file.read_text()
        required_vars = [
            "POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", 
            "NEO4J_HOST", "REDIS_HOST"
        ]
        
        for var in required_vars:
            if var in env_content:
                print(f"✅ {var} configured in .env")
            else:
                print(f"❌ {var} missing from .env")
                return False
    else:
        print("❌ .env file not found")
        return False
        
    # Check config.yaml if exists
    config_file = Path("config/config.yaml")
    if config_file.exists():
        print("✅ config/config.yaml exists")
    else:
        print("⚠️  config/config.yaml not found (using .env only)")
        
    return True

async def main():
    """Run all tests."""
    print("🚀 Tyra MCP Memory Server - Database Setup Test")
    print("=" * 60)
    
    results = []
    
    # Test configuration
    config_ok = await test_config_files()
    results.append(("Configuration", config_ok))
    
    # Test databases
    postgres_ok = await test_postgresql_setup()
    results.append(("PostgreSQL", postgres_ok))
    
    redis_ok = await test_redis_setup()
    results.append(("Redis", redis_ok))
    
    neo4j_ok = await test_neo4j_setup()
    results.append(("Neo4j", neo4j_ok))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Your database setup is ready.")
        print("\nNext steps:")
        print("1. Install embedding models (see INSTALLATION.md Step 5)")
        print("2. Start the server: python -m src.mcp.server")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Run ./scripts/setup_existing_databases.sh to fix issues.")
        
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)