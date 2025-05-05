import asyncio
import threading
import time
from collections import namedtuple
from enum import Enum, IntEnum
from typing import AsyncGenerator, List, Optional, Tuple

import redis

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server.abstract_server import \
    LookupServerInterface  # noqa: E501
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class _WorkPriority(IntEnum):
    LOOKUP = 0  # Highest priority
    UPDATE = 1  # Lowest priority


class _Op(Enum):
    INSERT = 1
    DELETE = 2
    LOOKUP = 3


_WorkItem = namedtuple('_WorkItem', ['key', 'op', 'fut'])


class RedisLookupServer(LookupServerInterface):

    def __init__(self, config: LMCacheEngineConfig):
        assert config.distributed_url is not None
        self.distributed_url: str = config.distributed_url

        self.url = config.lookup_url
        assert self.url is not None
        host, port = self.url.split(":")
        self.host = host
        self.port = int(port)

        self.batch_timeout = config.lookup_batch_timeout
        self.batch_size = config.lookup_batch_size
        self.lookup_timeout = config.lookup_timeout

        self.connection = redis.Redis(host=self.host,
                                      port=self.port,
                                      socket_timeout=self.lookup_timeout,
                                      decode_responses=True)
        logger.info(f"Connected to Redis lookup server at {host}:{port}")
        #decode_responses=False)

        self.queue: asyncio.PriorityQueue[tuple[_WorkPriority, _WorkItem]] = (
            asyncio.PriorityQueue(config.lookup_queue_size))
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._requester(), self.loop)

    async def _get_batch(self) -> AsyncGenerator[_WorkItem, None]:
        """
        Batch work items as an async generator.

        :yields: WorkItem's one by one as they are retrieved.
        """
        # wait for first item in batch
        _, item = await self.queue.get()
        yield item

        if item.op == _Op.LOOKUP:
            timeout = 0.0  # lookups should not wait
        else:
            timeout = self.batch_timeout

        for _ in range(self.batch_size - 1):
            try:
                _, item = await asyncio.wait_for(self.queue.get(),
                                                 timeout=timeout)
                yield item
                if timeout > 0 and item.op == _Op.LOOKUP:
                    timeout = 0  # lookups should not wait
            except asyncio.TimeoutError:
                break

    def _add_to_pipeline(self, pipe: redis.client.Pipeline, item: _WorkItem):
        """
        Add a request to a redis pipeline (batch of requests).

        :param redis.client.Pipeline pipe: The redis pipeline to add to.

        :param _WorkItem item: The item defining the request we want to add.
        """
        match item.op:
            case _Op.INSERT:
                pipe.hset(self._get_indexing_key(item.key),
                          self.distributed_url,
                          self._get_indexing_metadata(item.key))
            case _Op.DELETE:
                pipe.hdel(self._get_indexing_key(item.key),
                          self.distributed_url)
            case _Op.LOOKUP:
                pipe.hgetall(self._get_indexing_key(item.key))

    async def _requester(self):
        """
        _requester batches requests to be sent to redis
        """
        while True:
            items = []
            with self.connection.pipeline() as pipe:
                async for item in self._get_batch():
                    self._add_to_pipeline(pipe, item)
                    items.append(item)
                logger.debug(f"Sending a batch of {len(items)} requests")
                results = pipe.execute(raise_on_error=False)
                logger.debug("Batch results are ready")

            for item, result in zip(items, results):
                if isinstance(result, Exception):
                    item.fut.set_exception(result)
                else:
                    item.fut.set_result(result)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def close(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()
        self.connection.close()

    def _get_indexing_key(self, key: CacheEngineKey) -> str:
        return f"{key.model_name}@{key.chunk_hash}"

    def _get_indexing_metadata(self, key: CacheEngineKey) -> str:
        return f"{key.fmt}@{key.world_size}@{key.worker_id}@{time.time()}"

    def _extract_cache_keys_componenets(self, metadata: str) -> List[str]:
        return metadata.split("@")[:-1]

    def lookup(self, key: CacheEngineKey) -> Optional[Tuple[str, int]]:
        """
        Perform lookup in the lookup server.
        """
        logger.debug("Call to lookup in lookup server")
        fut = self.loop.create_future()
        item = _WorkItem(key, _Op.LOOKUP, fut)
        asyncio.run_coroutine_threadsafe(
            self.queue.put((_WorkPriority.LOOKUP, item)), self.loop)

        cfut = asyncio.run_coroutine_threadsafe(_wait_for_future(fut),
                                                self.loop)

        result = None
        try:
            result = cfut.result(self.lookup_timeout)
        except TimeoutError:
            logger.warning("Timeout while waiting for lookup")
            fut.cancel()
        except Exception as exc:
            logger.warning(f"Got exception while doing lookup: {exc}")

        if not result:
            return None

        logger.debug(f"KV cache lives on {result}")
        comps = self._extract_cache_keys_componenets(
            self._get_indexing_metadata(key))

        for md_key, md_value in result.items():
            if comps == self._extract_cache_keys_componenets(md_value):
                url = md_key
                host, port = url.split(":")
                return host, int(port)
        return None

    def insert(self, key: CacheEngineKey):
        """
        Perform insert in the lookup server.
        """
        logger.debug("Call to insert in lookup server")
        fut = self.loop.create_future()
        item = _WorkItem(key, _Op.INSERT, fut)
        fut.add_done_callback(_log_result)
        asyncio.run_coroutine_threadsafe(
            self.queue.put((_WorkPriority.UPDATE, item)), self.loop)

    def remove(self, key: CacheEngineKey):
        """
        Perform remove in the lookup server.
        """
        logger.debug("Call to remove in lookup server")
        fut = self.loop.create_future()
        item = _WorkItem(key, _Op.DELETE, fut)
        fut.add_done_callback(_log_result)
        asyncio.run_coroutine_threadsafe(
            self.queue.put((_WorkPriority.UPDATE, item)), self.loop)

    def batched_remove(self, keys: List[CacheEngineKey]):
        """
        Perform batched remove in the lookup server.
        """
        logger.debug("Call to batched remove in lookup server")
        for key in keys:
            self.remove(key)


async def _wait_for_future(fut: asyncio.Future):
    return await fut


def _log_result(fut: asyncio.Future):
    try:
        result = fut.result()
        logger.debug(f"Result: {result}")
    except Exception as e:
        logger.warning(f"Got exception: {e}")
