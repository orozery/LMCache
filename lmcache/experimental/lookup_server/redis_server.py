import inspect
import time
from typing import List, Optional, Tuple

import redis

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server.abstract_server import \
    LookupServerInterface  # noqa: E501
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


# TODO (Jiayi): Batching is needed for Redis lookup server.
class RedisLookupServer(LookupServerInterface):

    def __init__(self, config: LMCacheEngineConfig):
        self.distributed_url = config.distributed_url
        assert self.distributed_url is not None

        self.url = config.lookup_url
        assert self.url is not None
        host, port = self.url.split(":")
        self.host = host
        self.port = int(port)

        self.connection = redis.Redis(host=host,
                                      port=port,
                                      decode_responses=True)
        logger.info(f"Connected to Redis lookup server at {host}:{port}")
        #decode_responses=False)

    def _get_indexing_key(self, key: CacheEngineKey):
        return f"{key.model_name}@{key.chunk_hash}"

    def _get_indexing_metadata(self, key: CacheEngineKey):
        return f"{key.fmt}@{key.world_size}@{key.worker_id}@{time.time()}"

    def _extract_cache_keys_componenets(self, indexing_metadata: str):
        return indexing_metadata.split("@")[:-1]

    def lookup(self, key: CacheEngineKey) -> Optional[Tuple[str, int]]:
        """
        Perform lookup in the lookup server.
        """
        logger.debug("Call to lookup in lookup server")
        result = self.connection.hgetall(self._get_indexing_key(key))
        logger.debug(f"KV cache lives on {result}")
        if not result:
            return None
        comps = self._extract_cache_keys_componenets(
            self._get_indexing_metadata(key))
        assert not inspect.isawaitable(result)
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
        assert self.distributed_url is not None
        logger.debug("Call to insert in lookup server")
        self.connection.hset(self._get_indexing_key(key), self.distributed_url,
                             self._get_indexing_metadata(key))

    def remove(self, key: CacheEngineKey):
        """
        Perform remove in the lookup server.
        """
        logger.debug("Call to remove in lookup server")
        assert self.distributed_url is not None
        self.connection.hdel(self._get_indexing_key(key), self.distributed_url)

    def batched_remove(self, keys: List[CacheEngineKey]):
        """
        Perform batched remove in the lookup server.
        """
        logger.debug("Call to batched remove in lookup server")
        if not keys:
            return
        # TODO(Jiayi): We might need to cache the `str_keys` for performance.
        pipe = self.connection.pipeline()
        assert self.distributed_url is not None
        for key in keys:
            pipe.hdel(self._get_indexing_key(key), self.distributed_url)
        pipe.execute()
