import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Generator, List, Optional, Sequence, TypedDict

import psycopg
from databricks.sdk import WorkspaceClient
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

class CredentialConnection(psycopg.Connection):
    """Custom connection class that generates fresh OAuth tokens with caching."""

    workspace_client = None
    instance_name = None

    # Cache attributes
    _cached_credential = None
    _cache_timestamp = None
    _cache_duration = 3000  # 50 minutes in seconds (50 * 60)
    _cache_lock = Lock()


    @classmethod
    def connect(cls, conninfo="", **kwargs):
        """Override connect to inject OAuth token with 50-minute caching"""
        if cls.workspace_client is None or cls.instance_name is None:
            raise ValueError(
                "workspace_client and instance_name must be set on CredentialConnection class"
            )

        # Get cached or fresh credential and append the new password to kwargs
        credential_token = cls._get_cached_credential()
        kwargs["password"] = credential_token

        # Call the superclass's connect method with updated kwargs
        return super().connect(conninfo, **kwargs)

    @classmethod
    def _get_cached_credential(cls):
        """Get credential from cache or generate a new one if cache is expired"""
        with cls._cache_lock:
            current_time = time.time()

            # Check if we have a valid cached credential
            if (
                cls._cached_credential is not None
                and cls._cache_timestamp is not None
                and current_time - cls._cache_timestamp < cls._cache_duration
            ):
                return cls._cached_credential

            # Generate new credential
            credential = cls.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()), instance_names=[cls.instance_name]
            )

            # Cache the new credential
            cls._cached_credential = credential.token
            cls._cache_timestamp = current_time

            return cls._cached_credential

class DatabricksStateManager:
    """
    A comprehensive state management library for Databricks Lakebase PostgreSQL.
    
    Features:
    - Direct connection management with automatic credential rotation and caching
    - Thread-safe operations with proper connection handling
    - Configurable cache duration
    - Built-in PostgresSaver integration for LangGraph checkpointing
    - Support for client ID/secret authentication
    - Standard PostgresSaver checkpoint table management
    
    Note: PostgresSaver uses standard table names (checkpoints, checkpoint_blobs, 
    checkpoint_writes) and doesn't support custom table naming in the current version.
    """
    
    def __init__(
        self,
        lakebase_config: Dict[str, Any],
        workspace_client: Optional[WorkspaceClient] = None,
        token_cache_minutes: int = 50,
        connection_timeout: float = 30.0
    ):
        """
        Initialize the state manager.
        
        Args:
            lakebase_config: Dictionary containing:
                - instance_name: Lakebase instance name
                - conn_host: Database host
                - conn_db_name: Database name (default: 'databricks_postgres')
                - conn_ssl_mode: SSL mode (default: 'require')
                - client_id: Service Principal client ID (optional)
                - client_secret: Service Principal client secret (optional)
                - workspace_host: Databricks workspace URL (required if using client_id/secret)
            workspace_client: Databricks workspace client (creates new if None)
            token_cache_minutes: How long to cache OAuth tokens
            connection_timeout: Connection timeout in seconds
        """
        self.lakebase_config = lakebase_config
        self.connection_timeout = connection_timeout
        # Connection pool configuration
        self.pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
        self.pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
        self.pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))

        # Initialize workspace client based on provided config
        if workspace_client:
            self.workspace_client = workspace_client
        elif lakebase_config.get("client_id") and lakebase_config.get("client_secret"):
            # Use client ID and secret for authentication
            workspace_host = lakebase_config.get("workspace_host")
            if not workspace_host:
                raise ValueError("workspace_host is required when using client_id/client_secret authentication")
            
            self.workspace_client = WorkspaceClient(
                host=workspace_host,
                client_id=lakebase_config["client_id"],
                client_secret=lakebase_config["client_secret"]
            )
            logger.info("WorkspaceClient initialized with client_id/client_secret authentication")
        else:
            # Use default authentication (environment variables, etc.)
            self.workspace_client = WorkspaceClient()
            logger.info("WorkspaceClient initialized with default authentication")
        
        # Token caching
        self._cache_duration = token_cache_minutes * 60
        self._cached_credential = None
        self._cache_timestamp = None
        self._cache_lock = Lock()
        
        # Standard PostgresSaver table names (not configurable in current version)
        self.standard_tables = {
            "checkpoints": "checkpoints",
            "checkpoint_blobs": "checkpoint_blobs", 
            "checkpoint_writes": "checkpoint_writes"
        }
        
        # Connection parameters
        self.username = self._get_username()
        self.host = self.lakebase_config["conn_host"]
        self.database = self.lakebase_config.get("conn_db_name", "databricks_postgres")
        self.ssl_mode = self.lakebase_config.get("conn_ssl_mode", "require")
        self.conn_info = f"dbname={self.database} user={self.username} host={self.host} sslmode={self.ssl_mode}"
        
        self._is_initialized = False
        
        # Initialize the connection pool with rotating credentials
        self._connection_pool = self._create_rotating_pool()
        print("Connection pool initialised")
        
        logger.info(
            f"DatabricksStateManager initialized with direct connections "
            f"using standard PostgresSaver tables: {', '.join(self.standard_tables.values())}"
        )
    
    def _get_username(self) -> str:
        """Get the username for database connection"""
        # If using client_id/secret authentication, use the client_id as username
        if self.lakebase_config.get("client_id"):
            return self.lakebase_config["client_id"]
        
        # Otherwise, determine username from workspace client
        try:
            sp = self.workspace_client.current_service_principal.me()
            return sp.application_id
        except Exception:
            user = self.workspace_client.current_user.me()
            return user.user_name
        
    def _create_rotating_pool(self) -> ConnectionPool:
        """Create a connection pool that automatically rotates credentials with caching"""

        CredentialConnection.workspace_client = self.workspace_client
        CredentialConnection.instance_name = self.lakebase_config["instance_name"]
        # Token cache duration (in minutes, can be overridden via env var)
        cache_duration_minutes = int(os.getenv("DB_TOKEN_CACHE_MINUTES", "50"))
        CredentialConnection._cache_duration = cache_duration_minutes * 60
        # Create pool with custom connection class
        pool = ConnectionPool(
            conninfo=f"dbname={self.database} user={self.username} host={self.host} sslmode={self.ssl_mode}",
            connection_class=CredentialConnection,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            timeout=self.pool_timeout,
            open=True,
            kwargs={
                "autocommit": True,  # Required for the .setup() method to properly commit the checkpoint tables to the database
                "row_factory": dict_row,  # Required because the PostgresSaver implementation accesses database rows using dictionary-style syntax
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )

        self._test_connection(pool)

        return pool
    
    def _get_cached_credential(self):
        """Get credential from cache or generate a new one if cache is expired"""
        with self._cache_lock:
            current_time = time.time()
            
            # Check if we have a valid cached credential
            if (self._cached_credential is not None and 
                self._cache_timestamp is not None and 
                current_time - self._cache_timestamp < self._cache_duration):
                return self._cached_credential
            
            # Generate new credential
            credential = self.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[self.lakebase_config["instance_name"]]
            )
            
            # Cache the new credential
            self._cached_credential = credential.token
            self._cache_timestamp = current_time
            
            return self._cached_credential
    
    def _test_connection(self, pool: ConnectionPool) -> None:
        """Test the connection to ensure it works"""
        try:
            with pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            logger.info(
                f"Connection test successful "
                f"(token_cache={self._cache_duration / 60:.0f} minutes)"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect using conninfo: {self.conn_info} Error: {e}")
    
    @contextmanager
    def get_connection(self):
        """Context manager to get a connection from the pool"""
        with self._connection_pool.connection() as conn:
            yield conn
    
    def create_checkpointer(self, connection=None) -> PostgresSaver:
        """
        Create a PostgresSaver instance with standard table names.
        
        Args:
            connection: Optional connection to use. If None, uses connection from pool.
            
        Returns:
            PostgresSaver instance with standard table names
            
        Note:
            PostgresSaver uses hardcoded table names and doesn't support 
            custom table naming in the current version.
        """
        if connection is None:
            # This will be used within a context manager
            raise ValueError("Connection must be provided when creating checkpointer")
        
        # Create standard PostgresSaver - no table_name parameter available
        return PostgresSaver(connection)
    
    def get_table_info(self) -> Dict[str, str]:
        """
        Get information about the checkpoint tables being used.
        
        Returns:
            Dictionary with table information
        """
        return {
            "standard_tables": self.standard_tables,
            "database_name": self.lakebase_config.get("conn_db_name", "databricks_postgres"),
            "host": self.lakebase_config["conn_host"],
            "instance_name": self.lakebase_config["instance_name"],
            "note": "PostgresSaver uses standard hardcoded table names"
        }
    
    def list_tables(self) -> List[str]:
        """
        List all tables in the database to verify checkpoint tables exist.
        
        Returns:
            List of table names
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                return [row[0] for row in cur.fetchall()]
    
    def verify_checkpoint_tables(self) -> Dict[str, bool]:
        """
        Verify that the standard checkpoint tables exist.
        
        Returns:
            Dictionary mapping table names to existence status
        """
        try:
            tables = self.list_tables()
            return {
                table_name: table_name in tables
                for table_name in self.standard_tables.values()
            }
        except Exception as e:
            logger.error(f"Error verifying checkpoint tables: {e}")
            return {table_name: False for table_name in self.standard_tables.values()}
    
    def get_checkpoint_config(self, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get checkpoint configuration for LangGraph.
        
        Args:
            thread_id: Thread ID for conversation state. Generates new if None.
            
        Returns:
            Configuration dictionary for LangGraph checkpointing
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            
        return {"configurable": {"thread_id": thread_id}}
    def initialize_checkpoint_tables(
        self, 
        drop_existing: bool = False
    ) -> bool:
        """
        Initialize LangGraph checkpoint tables in PostgreSQL.
        
        Args:
            state_manager: DatabricksStateManager instance (handles auth & connection)
            drop_existing: If True, drop existing tables before creating (DANGER!)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("="*60)
            logger.info("LangGraph Checkpoint Database Initialization")
            logger.info("="*60)
            
            # Test connection using state manager
            logger.info("Testing connection via DatabricksStateManager...")
            try:
                with self.get_connection() as test_conn:
                    logger.info("✓ Database connection successful")
            except Exception as e:
                logger.error(f"✗ Database connection failed: {e}")
                return False
            
            # Drop existing tables if requested
            if drop_existing:
                logger.warning("⚠️  DROP_EXISTING=True - Deleting all checkpoint data!")
                response = input("Are you sure? Type 'yes' to confirm: ")
                if response.lower() != 'yes':
                    logger.info("Aborted by user")
                    return False
                
                logger.info("Dropping existing tables...")
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DROP TABLE IF EXISTS checkpoint_writes CASCADE")
                        cur.execute("DROP TABLE IF EXISTS checkpoint_blobs CASCADE")
                        cur.execute("DROP TABLE IF EXISTS checkpoints CASCADE")
                        conn.commit()
                        logger.info("✓ Existing tables dropped")
            
            # Initialize PostgresSaver (creates tables automatically)
            logger.info("Initializing PostgresSaver...")
            
            with self.get_connection() as conn:
                # Create PostgresSaver instance
                checkpointer = PostgresSaver(conn)
                
                # Call setup to create tables
                checkpointer.setup()
                
                logger.info("✓ PostgresSaver initialized")
                logger.info("✓ Tables created successfully")
            
            # Verify tables exist
            logger.info("\nVerifying table creation...")
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check for required tables
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes')
                        ORDER BY table_name
                    """)
                    
                    rows = cur.fetchall()
                    if rows and isinstance(rows[0], dict):
                        tables = [row['table_name'] for row in rows]
                    else:
                        tables = [row[0] for row in rows]
                    
                    expected_tables = ['checkpoint_blobs', 'checkpoint_writes', 'checkpoints']
                    
                    logger.info("\nTable Status:")
                    for table in expected_tables:
                        if table in tables:
                            logger.info(f"  ✓ {table}")
                        else:
                            logger.error(f"  ✗ {table} - MISSING!")
                            return False
                    
                    # Get row counts
                    logger.info("\nTable Statistics:")
                    for table in tables:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        row = cur.fetchone()
                        count = row['count'] if isinstance(row, dict) else row[0]
                        logger.info(f"  {table}: {count} rows")
            
            logger.info("\n" + "="*60)
            logger.info("✓ Initialization Complete!")
            logger.info("="*60)
            logger.info("\nYour agent is ready to use checkpoint persistence.")
            
            return True
            
        except Exception as e:
            logger.error(f"\n✗ Initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def close(self) -> None:
        """Close and cleanup resources (no pool to close in this version)"""
        # Clear cached credentials for security
        with self._cache_lock:
            self._cached_credential = None
            self._cache_timestamp = None
        logger.info("DatabricksStateManager closed and credentials cleared")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize_checkpoint_tables()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    