"""State management layer for Session objects and heartbeat statuses in Mosaic nodes."""

import logging
from typing import Dict, List, Optional, Tuple

from mosaic_config.config import MosaicConfig
from mosaic_config.state import (
    ReceiveHeartbeatStatus,
    SendHeartbeatStatus,
    Session,
)
from mosaic_config.state_utils import StateIdentifiers, read_state, save_state

logger = logging.getLogger(__name__)


class SessionStateManager:
    """
    Manages Session objects with automatic persistence.
    
    This class provides a consistent interface for interacting with Session objects
    and ensures that all manipulations trigger automatic saves to persistent storage.
    """

    def __init__(self, config: MosaicConfig):
        """
        Initialize the SessionStateManager and load existing sessions.
        
        Args:
            config: MosaicConfig instance with state_location configured
        """
        self._config = config
        self._sessions: List[Session] = []
        self._load_sessions()

    def _load_sessions(self) -> None:
        """
        Load sessions from persistent state on startup.
        
        If the pickle file exists, sessions are loaded. Otherwise, an empty list is initialized.
        """
        try:
            loaded_sessions = read_state(self._config, StateIdentifiers.SESSIONS, default=None)
            
            if isinstance(loaded_sessions, list):
                self._sessions = loaded_sessions
                logger.info(f"Loaded {len(self._sessions)} sessions from state")
            else:
                self._sessions = []
                logger.info("No sessions found in state, initializing empty list")
        except Exception as e:
            logger.warning(f"Error loading Sessions from state: {e}")
            self._sessions = []

    def _save_sessions(self) -> None:
        """
        Save the current sessions list to persistent state.
        
        This is called automatically after any manipulation operation.
        """
        try:
            save_state(self._config, self._sessions, StateIdentifiers.SESSIONS)
            logger.debug(f"Saved {len(self._sessions)} sessions to state")
        except Exception as e:
            logger.warning(f"Failed to save sessions state: {e}")
            raise

    def get_sessions(self) -> List[Session]:
        """
        Get a copy of the current sessions list.
        
        Returns:
            List of Session objects (a copy to prevent external modification)
        """
        return list(self._sessions)

    def get_session_by_id(self, session_id: str) -> Optional[Session]:
        """
        Get a session by its ID.
        
        Args:
            session_id: The ID of the session to retrieve
            
        Returns:
            Session object if found, None otherwise
        """
        for session in self._sessions:
            if session.id == session_id:
                return session
        return None

    def add_session(self, session: Session) -> None:
        """
        Add a session to the list and persist state.
        
        Args:
            session: Session instance to add
        """
        self._sessions.append(session)
        self._save_sessions()
        logger.debug(f"Added session with ID: {session.id}")

    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session from the list by ID and persist state.
        
        Args:
            session_id: ID of the session to remove
            
        Returns:
            True if session was found and removed, False otherwise
        """
        initial_count = len(self._sessions)
        self._sessions = [s for s in self._sessions if s.id != session_id]
        
        if len(self._sessions) < initial_count:
            self._save_sessions()
            logger.debug(f"Removed session with ID: {session_id}")
            return True
        else:
            logger.warning(f"Session with ID {session_id} not found")
            return False

    def update_session(self, session_id: str, **kwargs) -> bool:
        """
        Update a session's attributes and persist state.
        
        Args:
            session_id: ID of the session to update
            **kwargs: Attributes to update (e.g., status=SessionStatus.COMPLETE)
            
        Returns:
            True if session was found and updated, False otherwise
        """
        session = self.get_session_by_id(session_id)
        if session is None:
            logger.warning(f"Session with ID {session_id} not found for update")
            return False
        
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
            else:
                logger.warning(f"Session does not have attribute '{key}', skipping")
        
        self._save_sessions()
        logger.debug(f"Updated session with ID: {session_id}")
        return True

    def clear_sessions(self) -> None:
        """
        Clear all sessions and persist state.
        """
        self._sessions = []
        self._save_sessions()
        logger.debug("Cleared all sessions")


class HeartbeatStateManager:
    """
    Manages heartbeat status objects with automatic persistence.
    
    This class provides a consistent interface for interacting with heartbeat statuses
    and ensures that all manipulations can trigger automatic saves to persistent storage.
    """

    def __init__(self, config: MosaicConfig):
        """
        Initialize the HeartbeatStateManager and load existing heartbeat statuses.
        
        Args:
            config: MosaicConfig instance with state_location configured
        """
        self._config = config
        self._send_heartbeat_statuses: Dict[Tuple[str, int], SendHeartbeatStatus] = {}
        self._receive_heartbeat_statuses: Dict[Tuple[str, int], ReceiveHeartbeatStatus] = {}
        self._load_heartbeat_statuses()

    def _load_heartbeat_statuses(self) -> None:
        """
        Load heartbeat statuses from persistent state on startup.
        
        If the pickle files exist, statuses are loaded. Otherwise, empty dictionaries are initialized.
        """
        try:
            loaded_send = read_state(
                self._config, StateIdentifiers.SEND_HEARTBEAT_STATUSES, default=None
            )
            
            if isinstance(loaded_send, dict):
                self._send_heartbeat_statuses = loaded_send
                logger.info(f"Loaded {len(loaded_send)} send heartbeat statuses from state")
            else:
                self._send_heartbeat_statuses = {}
                logger.info("No send heartbeat statuses found in state, initializing empty dict")
        except Exception as e:
            logger.warning(f"Error loading send heartbeat statuses from state: {e}")
            self._send_heartbeat_statuses = {}
        
        try:
            loaded_receive = read_state(
                self._config, StateIdentifiers.RECEIVE_HEARTBEAT_STATUSES, default=None
            )
            
            if isinstance(loaded_receive, dict):
                self._receive_heartbeat_statuses = loaded_receive
                logger.info(f"Loaded {len(loaded_receive)} receive heartbeat statuses from state")
            else:
                self._receive_heartbeat_statuses = {}
                logger.info("No receive heartbeat statuses found in state, initializing empty dict")
        except Exception as e:
            logger.warning(f"Error loading receive heartbeat statuses from state: {e}")
            self._receive_heartbeat_statuses = {}

    def save_heartbeat_statuses(self) -> None:
        """
        Save the current heartbeat statuses to persistent state.
        
        This should be called periodically (e.g., after sending heartbeats) to persist state.
        """
        try:
            save_state(
                self._config,
                self._send_heartbeat_statuses,
                StateIdentifiers.SEND_HEARTBEAT_STATUSES,
            )
            save_state(
                self._config,
                self._receive_heartbeat_statuses,
                StateIdentifiers.RECEIVE_HEARTBEAT_STATUSES,
            )
            logger.debug(
                f"Saved {len(self._send_heartbeat_statuses)} send and "
                f"{len(self._receive_heartbeat_statuses)} receive heartbeat statuses to state"
            )
        except Exception as e:
            logger.warning(f"Failed to save heartbeat statuses state: {e}")
            raise

    def get_send_heartbeat_statuses(self) -> Dict[Tuple[str, int], SendHeartbeatStatus]:
        """
        Get a copy of the current send heartbeat statuses dictionary.
        
        Returns:
            Dictionary of send heartbeat statuses (a copy to prevent external modification)
        """
        return dict(self._send_heartbeat_statuses)

    def get_receive_heartbeat_statuses(self) -> Dict[Tuple[str, int], ReceiveHeartbeatStatus]:
        """
        Get a copy of the current receive heartbeat statuses dictionary.
        
        Returns:
            Dictionary of receive heartbeat statuses (a copy to prevent external modification)
        """
        return dict(self._receive_heartbeat_statuses)

    def get_send_heartbeat_status(
        self, host: str, heartbeat_port: int
    ) -> Optional[SendHeartbeatStatus]:
        """
        Get send heartbeat status for a specific peer.
        
        Args:
            host: Peer host address
            heartbeat_port: Peer heartbeat port
            
        Returns:
            SendHeartbeatStatus if found, None otherwise
        """
        key = (host, heartbeat_port)
        return self._send_heartbeat_statuses.get(key)

    def get_receive_heartbeat_status(
        self, host: str, heartbeat_port: int
    ) -> Optional[ReceiveHeartbeatStatus]:
        """
        Get receive heartbeat status for a specific peer.
        
        Args:
            host: Peer host address
            heartbeat_port: Peer heartbeat port
            
        Returns:
            ReceiveHeartbeatStatus if found, None otherwise
        """
        key = (host, heartbeat_port)
        return self._receive_heartbeat_statuses.get(key)

    def update_send_heartbeat_status(
        self, host: str, heartbeat_port: int, status: SendHeartbeatStatus
    ) -> None:
        """
        Update or create a send heartbeat status.
        
        Args:
            host: Peer host address
            heartbeat_port: Peer heartbeat port
            status: SendHeartbeatStatus object to store
        """
        key = (host, heartbeat_port)
        self._send_heartbeat_statuses[key] = status
        logger.debug(f"Updated send heartbeat status for {host}:{heartbeat_port}")

    def update_receive_heartbeat_status(
        self, host: str, heartbeat_port: int, status: ReceiveHeartbeatStatus
    ) -> None:
        """
        Update or create a receive heartbeat status.
        
        Args:
            host: Peer host address
            heartbeat_port: Peer heartbeat port
            status: ReceiveHeartbeatStatus object to store
        """
        key = (host, heartbeat_port)
        self._receive_heartbeat_statuses[key] = status
        logger.debug(f"Updated receive heartbeat status for {host}:{heartbeat_port}")

    def get_all_send_heartbeat_statuses(self) -> List[SendHeartbeatStatus]:
        """
        Get list of all send heartbeat statuses.
        
        Returns:
            List of SendHeartbeatStatus objects
        """
        return list(self._send_heartbeat_statuses.values())

    def get_all_receive_heartbeat_statuses(self) -> List[ReceiveHeartbeatStatus]:
        """
        Get list of all receive heartbeat statuses.
        
        Returns:
            List of ReceiveHeartbeatStatus objects
        """
        return list(self._receive_heartbeat_statuses.values())

    def clear_heartbeat_statuses(self) -> None:
        """
        Clear all heartbeat statuses and persist state.
        """
        self._send_heartbeat_statuses = {}
        self._receive_heartbeat_statuses = {}
        self.save_heartbeat_statuses()
        logger.debug("Cleared all heartbeat statuses")

