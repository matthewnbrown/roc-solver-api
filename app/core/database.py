"""
Thread-safe database management for the Captcha Solve API
"""

import sqlite3
import threading
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Thread-safe database manager with connection pooling and retry logic"""
    
    def __init__(self, db_path: str = "captcha_api.db", timeout: float = 30.0, retry_attempts: int = 3):
        self.db_path = db_path
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self._lock = threading.RLock()
        self._local = threading.local()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            
            # Set timeout for busy connections
            cursor.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
            
            # Create captcha_requests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS captcha_requests (
                    id TEXT PRIMARY KEY,
                    captcha_hash TEXT NOT NULL,
                    image_data BLOB,
                    image_filepath TEXT,
                    predicted_answer TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feedback_received BOOLEAN DEFAULT FALSE,
                    is_correct BOOLEAN
                )
            ''')
            
            # Create feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    is_correct BOOLEAN NOT NULL,
                    actual_answer TEXT,
                    feedback_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (request_id) REFERENCES captcha_requests (id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_captcha_requests_hash 
                ON captcha_requests(captcha_hash)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_captcha_requests_created_at 
                ON captcha_requests(created_at)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_feedback_request_id 
                ON feedback(request_id)
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with proper error handling and retry logic"""
        conn = None
        attempt = 0
        
        while attempt < self.retry_attempts:
            try:
                # Create connection with timeout
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=self.timeout,
                    check_same_thread=False
                )
                
                # Set connection options for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                
                # Enable row factory for easier data access
                conn.row_factory = sqlite3.Row
                
                yield conn
                return
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    attempt += 1
                    if attempt < self.retry_attempts:
                        wait_time = min(0.1 * (2 ** attempt), 1.0)  # Exponential backoff
                        logger.warning(f"Database locked, retrying in {wait_time}s (attempt {attempt}/{self.retry_attempts})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Database locked after {self.retry_attempts} attempts")
                        raise
                else:
                    logger.error(f"Database operational error: {e}")
                    raise
            except Exception as e:
                logger.error(f"Database connection error: {e}")
                raise
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception as e:
                        logger.error(f"Error closing database connection: {e}")
    
    def execute_with_retry(self, query: str, params: Tuple = (), fetch: bool = False):
        """Execute a query with retry logic"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute(query, params)
                
                if fetch:
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount
                    
            except sqlite3.IntegrityError as e:
                logger.error(f"Database integrity error: {e}")
                raise
            except Exception as e:
                logger.error(f"Database execution error: {e}")
                raise
    
    def insert_captcha_request(self, request_id: str, captcha_hash: str, 
                             image_data: bytes, predicted_answer: str, confidence: float, 
                             image_filepath: str = None) -> bool:
        """Insert a new captcha request"""
        query = '''
            INSERT INTO captcha_requests 
            (id, captcha_hash, image_data, image_filepath, predicted_answer, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        
        try:
            self.execute_with_retry(query, (request_id, captcha_hash, image_data, image_filepath, predicted_answer, confidence))
            logger.info(f"Inserted captcha request: {request_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert captcha request {request_id}: {e}")
            return False
    
    def get_captcha_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a captcha request by ID"""
        query = 'SELECT * FROM captcha_requests WHERE id = ?'
        
        try:
            results = self.execute_with_retry(query, (request_id,), fetch=True)
            if results:
                return dict(results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get captcha request {request_id}: {e}")
            return None
    
    def insert_feedback(self, request_id: str, is_correct: bool, actual_answer: Optional[str] = None) -> bool:
        """Insert feedback for a captcha request"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Start transaction
                cursor.execute("BEGIN TRANSACTION")
                
                # Check if request exists
                cursor.execute('SELECT id FROM captcha_requests WHERE id = ?', (request_id,))
                if not cursor.fetchone():
                    cursor.execute("ROLLBACK")
                    return False
                
                # Insert feedback
                cursor.execute('''
                    INSERT INTO feedback (request_id, is_correct, actual_answer)
                    VALUES (?, ?, ?)
                ''', (request_id, is_correct, actual_answer))
                
                # Update the original request
                cursor.execute('''
                    UPDATE captcha_requests 
                    SET feedback_received = TRUE, is_correct = ?
                    WHERE id = ?
                ''', (is_correct, request_id))
                
                # Commit transaction
                cursor.execute("COMMIT")
                logger.info(f"Inserted feedback for request: {request_id}")
                return True
                
            except Exception as e:
                cursor.execute("ROLLBACK")
                logger.error(f"Failed to insert feedback for {request_id}: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                stats = {}
                
                # Total requests
                cursor.execute('SELECT COUNT(*) FROM captcha_requests')
                stats['total_requests'] = cursor.fetchone()[0]
                
                # Requests with feedback
                cursor.execute('SELECT COUNT(*) FROM captcha_requests WHERE feedback_received = TRUE')
                stats['feedback_received'] = cursor.fetchone()[0]
                
                # Correct predictions
                cursor.execute('SELECT COUNT(*) FROM captcha_requests WHERE is_correct = TRUE')
                stats['correct_predictions'] = cursor.fetchone()[0]
                
                # Average confidence
                cursor.execute('SELECT AVG(confidence) FROM captcha_requests')
                avg_conf = cursor.fetchone()[0]
                stats['average_confidence'] = round(avg_conf, 4) if avg_conf else 0
                
                # Recent requests (last 24 hours)
                cursor.execute('''
                    SELECT COUNT(*) FROM captcha_requests 
                    WHERE created_at >= datetime('now', '-1 day')
                ''')
                stats['recent_requests_24h'] = cursor.fetchone()[0]
                
                # Calculate accuracy
                if stats['feedback_received'] > 0:
                    stats['accuracy_percentage'] = round(
                        (stats['correct_predictions'] / stats['feedback_received']) * 100, 2
                    )
                else:
                    stats['accuracy_percentage'] = 0
                
                return stats
                
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {
                    'total_requests': 0,
                    'feedback_received': 0,
                    'correct_predictions': 0,
                    'accuracy_percentage': 0,
                    'average_confidence': 0,
                    'recent_requests_24h': 0
                }
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old data (optional maintenance function)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Delete old captcha requests
                cursor.execute('''
                    DELETE FROM captcha_requests 
                    WHERE created_at < datetime('now', '-{} days')
                '''.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old records")
                
                return deleted_count
                
            except Exception as e:
                logger.error(f"Failed to cleanup old data: {e}")
                return 0

# Global database manager instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def init_database_manager(db_path: str = "captcha_api.db", timeout: float = 30.0, retry_attempts: int = 3) -> DatabaseManager:
    """Initialize the database manager"""
    global db_manager
    db_manager = DatabaseManager(db_path, timeout, retry_attempts)
    return db_manager
