"""
Retention purge job - Cleans up old completed trips
"""
import asyncio
import logging
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.services.trip_service import TripService

logger = logging.getLogger(__name__)


class PurgeJobScheduler:
    """
    Manages periodic purge of old completed trips
    
    In production, this would be integrated with APScheduler or similar.
    For now, provides manual trigger capability.
    """
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.is_running = False
    
    async def run_purge_job(self, db: AsyncSession) -> dict:
        """
        Execute purge job
        
        Args:
            db: Database session
            
        Returns:
            Purge statistics
        """
        if self.is_running:
            logger.warning("Purge job already running, skipping")
            return {"status": "skipped", "reason": "already_running"}
        
        self.is_running = True
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting purge job with {self.retention_days}-day retention")
            
            service = TripService(db)
            deleted_count = await service.purge_old_completed_trips(self.retention_days)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"Purge job completed: deleted {deleted_count} trips in {duration:.2f}s"
            )
            
            return {
                "status": "success",
                "deleted_count": deleted_count,
                "duration_seconds": duration,
                "retention_days": self.retention_days,
                "timestamp": start_time.isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Purge job failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": start_time.isoformat(),
            }
        finally:
            self.is_running = False


# Global scheduler instance
_purge_scheduler = PurgeJobScheduler(retention_days=30)


def get_purge_scheduler() -> PurgeJobScheduler:
    """Get global purge scheduler instance"""
    return _purge_scheduler


async def run_purge_job_once(retention_days: int = 30) -> dict:
    """
    Convenience function to run purge job once
    
    Args:
        retention_days: Retention period
        
    Returns:
        Purge statistics
    """
    async for db in get_db():
        scheduler = PurgeJobScheduler(retention_days=retention_days)
        result = await scheduler.run_purge_job(db)
        return result
