"""asyncpg 및 pgvector를 사용한 데이터베이스 연결 관리."""

from typing import Optional
import asyncpg
from pgvector.asyncpg import register_vector

from src.config import settings

# 전역 연결 풀
_pool: Optional[asyncpg.Pool] = None


async def get_db_pool() -> asyncpg.Pool:
    """데이터베이스 연결 풀을 가져오거나 생성합니다."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=settings.database_host,
            port=settings.database_port,
            database=settings.database_name,
            user=settings.database_user,
            password=settings.database_password or None,
            min_size=1,
            max_size=10,
            init=_init_connection,
        )
    return _pool


async def _init_connection(conn: asyncpg.Connection) -> None:
    """pgvector 확장을 사용하여 연결을 초기화합니다."""
    await register_vector(conn)


async def close_db_pool() -> None:
    """데이터베이스 연결 풀을 닫습니다."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def get_connection() -> asyncpg.Connection:
    """풀에서 연결을 가져옵니다."""
    pool = await get_db_pool()
    return await pool.acquire()


async def release_connection(conn: asyncpg.Connection) -> None:
    """연결을 풀로 반환합니다."""
    pool = await get_db_pool()
    await pool.release(conn)
