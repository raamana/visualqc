"""
Lightweight SQLite-based ratings storage with immediate checkpointing.

This module provides persistent storage for ratings that survives crashes.
"""

import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


class RatingsDatabase:
    """
    SQLite database for storing ratings with immediate checkpointing.
    
    Provides ACID transactions and crash-safe writes.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the ratings database.
        
        Parameters
        ----------
        db_path : Path
            Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Ratings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    unit_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    module_type TEXT,
                    vis_type TEXT,
                    PRIMARY KEY (unit_id, timestamp)
                )
            """)
            
            # Session metadata table for command line and versions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY,
                    command_line TEXT NOT NULL,
                    visualqc_version TEXT,
                    python_version TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
            """)
            
            # Index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_unit_id 
                ON ratings(unit_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON ratings(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id 
                ON ratings(session_id)
            """)
            
            conn.commit()
    
    def save_rating(
        self,
        unit_id: str,
        rating: str,
        notes: str = '',
        session_id: Optional[str] = None,
        module_type: Optional[str] = None,
        vis_type: Optional[str] = None
    ) -> bool:
        """
        Save a rating immediately to disk (checkpoint).
        
        This is called every time a rating is captured to ensure
        no work is lost in case of crash.
        
        Parameters
        ----------
        unit_id : str
            Subject/session/run identifier
        rating : str
            Rating value(s) - can be joined with '+'
        notes : str
            User notes
        session_id : str, optional
            Session identifier for tracking
        module_type : str, optional
            Module type (e.g., 'alignment', 'freesurfer')
        vis_type : str, optional
            Visualization type
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            timestamp = datetime.now().isoformat()
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO ratings 
                    (unit_id, rating, notes, timestamp, session_id, module_type, vis_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (unit_id, rating, notes, timestamp, session_id, module_type, vis_type))
                conn.commit()
            
            # Don't log success - quiet on success, only log errors
            return True
            
        except Exception as e:
            logger.error(f"Failed to checkpoint rating for {unit_id}: {e}")
            return False
    
    def get_latest_rating(self, unit_id: str) -> Optional[Tuple[str, str]]:
        """
        Get the most recent rating for a unit.
        
        Parameters
        ----------
        unit_id : str
            Subject/session/run identifier
            
        Returns
        -------
        tuple or None
            (rating, notes) or None if not found
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT rating, notes 
                    FROM ratings 
                    WHERE unit_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (unit_id,))
                result = cursor.fetchone()
                return result if result else None
        except Exception as e:
            logger.error(f"Failed to get rating for {unit_id}: {e}")
            return None
    
    def get_all_ratings(self) -> Dict[str, Tuple[str, str]]:
        """
        Get all latest ratings (one per unit_id).
        
        Returns
        -------
        dict
            {unit_id: (rating, notes)}
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT unit_id, rating, notes 
                    FROM ratings r1
                    WHERE timestamp = (
                        SELECT MAX(timestamp) 
                        FROM ratings r2 
                        WHERE r2.unit_id = r1.unit_id
                    )
                """)
                return {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get all ratings: {e}")
            return {}
    
    def save_session_metadata(
        self,
        session_id: str,
        command_line: str,
        visualqc_version: Optional[str] = None,
        python_version: Optional[str] = None
    ) -> bool:
        """
        Save session metadata (command line and versions).
        
        Parameters
        ----------
        session_id : str
            Session identifier
        command_line : str
            Full command line (after parsing/validation)
        visualqc_version : str, optional
            VisualQC version
        python_version : str, optional
            Python version
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            import sys
            
            created_at = datetime.now().isoformat()
            
            # Get versions if not provided
            if visualqc_version is None:
                try:
                    from visualqc import __version__
                    visualqc_version = __version__
                except ImportError:
                    visualqc_version = "unknown"
            
            if python_version is None:
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO session_metadata
                    (session_id, command_line, visualqc_version, 
                     python_version, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, command_line, visualqc_version, 
                      python_version, created_at, created_at))
                conn.commit()
            
            logger.info(f"Saved session metadata for {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")
            return False
    
    def get_session_metadata(self, session_id: str) -> Optional[dict]:
        """
        Get session metadata.
        
        Parameters
        ----------
        session_id : str
            Session identifier
            
        Returns
        -------
        dict or None
            Session metadata dictionary
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT command_line, visualqc_version, 
                           python_version, created_at, updated_at
                    FROM session_metadata
                    WHERE session_id = ?
                """, (session_id,))
                result = cursor.fetchone()
                if result:
                    return {
                        'command_line': result[0],
                        'visualqc_version': result[1],
                        'python_version': result[2],
                        'created_at': result[3],
                        'updated_at': result[4]
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get session metadata: {e}")
            return None
    
    def get_all_session_ids(self) -> list:
        """Get all unique session IDs from ratings."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT session_id 
                    FROM ratings 
                    WHERE session_id IS NOT NULL
                    ORDER BY session_id DESC
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get session IDs: {e}")
            return []
    
    def export_to_json(self, json_path: Path, pretty: bool = True, 
                       include_metadata: bool = True) -> bool:
        """
        Export ratings to JSON with comprehensive metadata.
        
        Parameters
        ----------
        json_path : Path
            Path to output JSON file
        pretty : bool
            Whether to format JSON with indentation
        include_metadata : bool
            Whether to include session metadata (command, flags, etc.)
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            ratings = self.get_all_ratings()
            
            # Get database statistics
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT unit_id) as total,
                           MIN(timestamp) as first_rating,
                           MAX(timestamp) as last_rating,
                           COUNT(DISTINCT session_id) as num_sessions
                    FROM ratings
                """)
                stats = cursor.fetchone()
            
            # Build metadata
            metadata = {
                "exported_at": datetime.now().isoformat(),
                "total_ratings": len(ratings),
                "database_path": str(self.db_path),
                "first_rating": stats[1] if stats and stats[1] else None,
                "last_rating": stats[2] if stats and stats[2] else None,
                "num_sessions": stats[3] if stats and stats[3] else 0,
            }
            
            # Add session metadata if requested
            if include_metadata:
                session_ids = self.get_all_session_ids()
                sessions_metadata = []
                for session_id in session_ids:
                    session_meta = self.get_session_metadata(session_id)
                    if session_meta:
                        sessions_metadata.append(session_meta)
                metadata["sessions"] = sessions_metadata
            
            # Build ratings with full details
            ratings_data = {}
            with sqlite3.connect(str(self.db_path)) as conn:
                for unit_id, (rating, notes) in ratings.items():
                    # Get latest rating details including timestamp and session
                    cursor = conn.execute("""
                        SELECT timestamp, session_id
                        FROM ratings
                        WHERE unit_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (unit_id,))
                    result = cursor.fetchone()
                    
                    ratings_data[unit_id] = {
                        "rating": rating,
                        "notes": notes,
                        "timestamp": result[0] if result else None,
                        "session_id": result[1] if result else None,
                    }
            
            export_data = {
                "metadata": metadata,
                "ratings": ratings_data
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(export_data, f, ensure_ascii=False)
            
            logger.info(f"Exported {len(ratings)} ratings with metadata to {json_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False
    
    def get_json_path(self) -> Path:
        """Get the JSON export path (same location as database)."""
        return self.db_path.with_suffix('.json')
    
    def is_json_stale(self) -> bool:
        """
        Check if JSON export is stale compared to database.
        
        Returns True if:
        - JSON doesn't exist
        - JSON is older than database
        - JSON has fewer ratings than database
        """
        json_path = self.get_json_path()
        
        # JSON doesn't exist
        if not json_path.exists():
            return True
        
        # Check timestamps
        json_mtime = json_path.stat().st_mtime
        db_mtime = self.db_path.stat().st_mtime
        
        if db_mtime > json_mtime:
            return True  # Database is newer
        
        # Check rating counts
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                json_count = len(json_data.get('ratings', {}))
        except Exception:
            return True  # Can't read JSON, consider it stale
        
        db_count = len(self.get_all_ratings())
        
        if db_count > json_count:
            return True  # Database has more ratings
        
        return False  # JSON is up to date
    
    def export_json_if_needed(self, force: bool = False) -> bool:
        """
        Export to JSON if it's stale or if forced.
        
        Parameters
        ----------
        force : bool
            Force export even if JSON is up to date
            
        Returns
        -------
        bool
            True if export was performed, False if skipped
        """
        if force or self.is_json_stale():
            json_path = self.get_json_path()
            return self.export_to_json(json_path)
        return False
    
    def export_to_csv(self, csv_path: Path) -> bool:
        """
        Export ratings to CSV for backward compatibility (read-only).
        
        Parameters
        ----------
        csv_path : Path
            Path to output CSV file
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            ratings = self.get_all_ratings()
            lines = [
                f"{unit_id},{rating},{notes}"
                for unit_id, (rating, notes) in ratings.items()
            ]
            
            with open(csv_path, 'w') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Exported {len(ratings)} ratings to CSV: {csv_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False
    
    def load_from_json(self, json_path: Path) -> Dict[str, Tuple[str, str]]:
        """
        Load ratings from JSON file (for backward compatibility).
        
        Parameters
        ----------
        json_path : Path
            Path to JSON file
            
        Returns
        -------
        dict
            {unit_id: (rating, notes)}
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ratings = {}
            for unit_id, rating_data in data.get('ratings', {}).items():
                if isinstance(rating_data, dict):
                    rating = rating_data.get('rating', '')
                    notes = rating_data.get('notes', '')
                else:
                    # Handle old format (if any)
                    rating = rating_data
                    notes = ''
                ratings[unit_id] = (rating, notes)
            
            logger.info(f"Loaded {len(ratings)} ratings from JSON: {json_path}")
            return ratings
        except Exception as e:
            logger.error(f"Failed to load from JSON: {e}")
            return {}

