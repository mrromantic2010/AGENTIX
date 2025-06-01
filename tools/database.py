"""
Database Tool for Agentix agents.

This tool provides database operations including:
- SQL query execution
- Database connection management
- Transaction handling
- Result formatting and validation
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import json

from .base import BaseTool, ToolConfig, ToolResult, ToolStatus


class DatabaseConfig(ToolConfig):
    """Configuration for database tool."""
    
    # Connection settings
    database_type: str = "postgresql"  # postgresql, mysql, sqlite, mongodb
    host: str = "localhost"
    port: int = 5432
    database: str
    username: str
    password: str
    
    # Connection pool settings
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: int = 30
    
    # Query settings
    max_query_time: int = 60
    max_result_rows: int = 10000
    enable_transactions: bool = True
    
    # Security settings
    allowed_operations: List[str] = Field(default_factory=lambda: ["SELECT", "INSERT", "UPDATE", "DELETE"])
    blocked_tables: List[str] = Field(default_factory=list)
    require_where_clause: bool = True  # For UPDATE/DELETE operations
    
    @validator('database_type')
    def validate_database_type(cls, v):
        allowed_types = ["postgresql", "mysql", "sqlite", "mongodb"]
        if v not in allowed_types:
            raise ValueError(f"database_type must be one of {allowed_types}")
        return v


class QueryResult(BaseModel):
    """Database query result."""
    
    success: bool
    rows_affected: int = 0
    data: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    execution_time: float = 0.0
    query: str = ""
    error: Optional[str] = None


class DatabaseTool(BaseTool):
    """
    Database Tool for performing database operations.
    
    This tool provides:
    - Multi-database support (PostgreSQL, MySQL, SQLite, MongoDB)
    - Connection pooling and management
    - Query validation and security checks
    - Transaction support
    - Result formatting and pagination
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.db_config = config
        self.connection_pool = None
        self.connection = None
        
        # Initialize database-specific components
        self._setup_database_driver()
    
    def _setup_database_driver(self):
        """Setup database-specific driver and connection."""
        try:
            if self.db_config.database_type == "postgresql":
                # In production, use asyncpg
                self.logger.info("PostgreSQL driver configured")
            elif self.db_config.database_type == "mysql":
                # In production, use aiomysql
                self.logger.info("MySQL driver configured")
            elif self.db_config.database_type == "sqlite":
                # In production, use aiosqlite
                self.logger.info("SQLite driver configured")
            elif self.db_config.database_type == "mongodb":
                # In production, use motor (async MongoDB driver)
                self.logger.info("MongoDB driver configured")
        except Exception as e:
            self.logger.error(f"Failed to setup database driver: {str(e)}")
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute database operation with the given parameters."""
        
        operation = parameters.get('operation', 'query')
        query = parameters.get('query', '')
        params = parameters.get('params', [])
        
        if not query:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error="Query is required",
                tool_name=self.name,
                tool_version=self.version
            )
        
        try:
            # Validate query security
            if not self._validate_query_security(query, operation):
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error="Query failed security validation",
                    tool_name=self.name,
                    tool_version=self.version
                )
            
            # Execute based on operation type
            if operation == "query":
                result = await self._execute_query(query, params)
            elif operation == "transaction":
                queries = parameters.get('queries', [])
                result = await self._execute_transaction(queries)
            elif operation == "batch":
                queries = parameters.get('queries', [])
                result = await self._execute_batch(queries)
            else:
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error=f"Unknown operation: {operation}",
                    tool_name=self.name,
                    tool_version=self.version
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS if result.success else ToolStatus.FAILURE,
                data=result.dict(),
                error=result.error,
                tool_name=self.name,
                tool_version=self.version
            )
            
        except Exception as e:
            self.logger.error(f"Database operation failed: {str(e)}")
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"Database operation failed: {str(e)}",
                tool_name=self.name,
                tool_version=self.version
            )
    
    async def _execute_query(self, query: str, params: List[Any] = None) -> QueryResult:
        """Execute a single database query."""
        start_time = datetime.now()
        
        try:
            # This is a placeholder implementation
            # In production, use actual database drivers
            
            if self.db_config.database_type == "postgresql":
                result = await self._execute_postgresql_query(query, params)
            elif self.db_config.database_type == "mysql":
                result = await self._execute_mysql_query(query, params)
            elif self.db_config.database_type == "sqlite":
                result = await self._execute_sqlite_query(query, params)
            elif self.db_config.database_type == "mongodb":
                result = await self._execute_mongodb_query(query, params)
            else:
                raise ValueError(f"Unsupported database type: {self.db_config.database_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            result.query = query
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                query=query
            )
    
    async def _execute_postgresql_query(self, query: str, params: List[Any] = None) -> QueryResult:
        """Execute PostgreSQL query."""
        # Placeholder implementation
        # In production, use asyncpg:
        # async with self.connection_pool.acquire() as conn:
        #     result = await conn.fetch(query, *params)
        
        self.logger.info(f"Executing PostgreSQL query: {query[:100]}...")
        
        # Simulate query execution
        await asyncio.sleep(0.1)
        
        # Mock result based on query type
        if query.strip().upper().startswith('SELECT'):
            return QueryResult(
                success=True,
                data=[
                    {"id": 1, "name": "Sample Record 1", "created_at": datetime.now().isoformat()},
                    {"id": 2, "name": "Sample Record 2", "created_at": datetime.now().isoformat()}
                ],
                columns=["id", "name", "created_at"],
                rows_affected=2
            )
        else:
            return QueryResult(
                success=True,
                rows_affected=1
            )
    
    async def _execute_mysql_query(self, query: str, params: List[Any] = None) -> QueryResult:
        """Execute MySQL query."""
        # Placeholder implementation
        self.logger.info(f"Executing MySQL query: {query[:100]}...")
        await asyncio.sleep(0.1)
        
        return QueryResult(
            success=True,
            data=[{"result": "MySQL query executed"}],
            columns=["result"],
            rows_affected=1
        )
    
    async def _execute_sqlite_query(self, query: str, params: List[Any] = None) -> QueryResult:
        """Execute SQLite query."""
        # Placeholder implementation
        self.logger.info(f"Executing SQLite query: {query[:100]}...")
        await asyncio.sleep(0.05)
        
        return QueryResult(
            success=True,
            data=[{"result": "SQLite query executed"}],
            columns=["result"],
            rows_affected=1
        )
    
    async def _execute_mongodb_query(self, query: str, params: List[Any] = None) -> QueryResult:
        """Execute MongoDB query."""
        # Placeholder implementation
        # In production, parse query as MongoDB operation
        self.logger.info(f"Executing MongoDB query: {query[:100]}...")
        await asyncio.sleep(0.1)
        
        return QueryResult(
            success=True,
            data=[{"_id": "507f1f77bcf86cd799439011", "name": "Sample Document"}],
            columns=["_id", "name"],
            rows_affected=1
        )
    
    async def _execute_transaction(self, queries: List[Dict[str, Any]]) -> QueryResult:
        """Execute multiple queries in a transaction."""
        if not self.db_config.enable_transactions:
            return QueryResult(
                success=False,
                error="Transactions are disabled"
            )
        
        start_time = datetime.now()
        total_rows_affected = 0
        all_data = []
        
        try:
            # Begin transaction
            self.logger.info(f"Starting transaction with {len(queries)} queries")
            
            for i, query_info in enumerate(queries):
                query = query_info.get('query', '')
                params = query_info.get('params', [])
                
                if not self._validate_query_security(query, 'query'):
                    raise Exception(f"Query {i+1} failed security validation")
                
                result = await self._execute_query(query, params)
                if not result.success:
                    raise Exception(f"Query {i+1} failed: {result.error}")
                
                total_rows_affected += result.rows_affected
                all_data.extend(result.data)
            
            # Commit transaction
            self.logger.info("Transaction committed successfully")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                success=True,
                data=all_data,
                rows_affected=total_rows_affected,
                execution_time=execution_time,
                query=f"Transaction with {len(queries)} queries"
            )
            
        except Exception as e:
            # Rollback transaction
            self.logger.error(f"Transaction failed, rolling back: {str(e)}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                success=False,
                error=f"Transaction failed: {str(e)}",
                execution_time=execution_time,
                query=f"Failed transaction with {len(queries)} queries"
            )
    
    async def _execute_batch(self, queries: List[Dict[str, Any]]) -> QueryResult:
        """Execute multiple queries in batch (non-transactional)."""
        start_time = datetime.now()
        results = []
        total_rows_affected = 0
        
        for i, query_info in enumerate(queries):
            query = query_info.get('query', '')
            params = query_info.get('params', [])
            
            try:
                if not self._validate_query_security(query, 'query'):
                    result = QueryResult(
                        success=False,
                        error=f"Query {i+1} failed security validation",
                        query=query
                    )
                else:
                    result = await self._execute_query(query, params)
                
                results.append(result.dict())
                if result.success:
                    total_rows_affected += result.rows_affected
                
            except Exception as e:
                result = QueryResult(
                    success=False,
                    error=str(e),
                    query=query
                )
                results.append(result.dict())
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResult(
            success=True,
            data=results,
            rows_affected=total_rows_affected,
            execution_time=execution_time,
            query=f"Batch execution of {len(queries)} queries"
        )
    
    def _validate_query_security(self, query: str, operation: str) -> bool:
        """Validate query for security compliance."""
        query_upper = query.strip().upper()
        
        # Check allowed operations
        query_operation = query_upper.split()[0] if query_upper else ""
        if query_operation not in self.db_config.allowed_operations:
            self.logger.warning(f"Operation '{query_operation}' not allowed")
            return False
        
        # Check for blocked tables
        for blocked_table in self.db_config.blocked_tables:
            if blocked_table.upper() in query_upper:
                self.logger.warning(f"Access to blocked table '{blocked_table}' attempted")
                return False
        
        # Require WHERE clause for UPDATE/DELETE
        if self.db_config.require_where_clause:
            if query_operation in ["UPDATE", "DELETE"] and "WHERE" not in query_upper:
                self.logger.warning(f"{query_operation} query without WHERE clause")
                return False
        
        # Basic SQL injection protection
        dangerous_patterns = [
            "DROP TABLE", "DROP DATABASE", "TRUNCATE", "ALTER TABLE",
            "CREATE USER", "GRANT", "REVOKE", "--", "/*", "*/"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in query_upper:
                self.logger.warning(f"Dangerous pattern '{pattern}' detected in query")
                return False
        
        return True
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate database operation parameters."""
        operation = parameters.get('operation', 'query')
        
        if operation not in ['query', 'transaction', 'batch']:
            return False
        
        if operation == 'query':
            query = parameters.get('query')
            if not query or not isinstance(query, str):
                return False
        
        elif operation in ['transaction', 'batch']:
            queries = parameters.get('queries')
            if not queries or not isinstance(queries, list):
                return False
            
            for query_info in queries:
                if not isinstance(query_info, dict) or 'query' not in query_info:
                    return False
        
        return True
    
    async def close(self):
        """Close database connections."""
        if self.connection_pool:
            # Close connection pool
            self.logger.info("Closing database connection pool")
            # await self.connection_pool.close()
        
        if self.connection:
            # Close single connection
            self.logger.info("Closing database connection")
            # await self.connection.close()
