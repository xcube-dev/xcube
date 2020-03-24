import os
import shutil
import unittest

from xcube.util.cache import CacheStore, Cache, MemoryCacheStore, FileCacheStore, parse_mem_size


class MemoryCacheStoreTest(unittest.TestCase):
    def setUp(self):
        self.cache_store = MemoryCacheStore()
        stored_value_a, size_a = self.cache_store.store_value('a', 42)
        stored_value_b, size_b = self.cache_store.store_value('b', True)
        stored_value_c, size_c = self.cache_store.store_value('c', "S" * 256)
        stored_value_d, size_d = self.cache_store.store_value('d', 3.14)
        self.stored_value_a = stored_value_a
        self.stored_value_b = stored_value_b
        self.stored_value_c = stored_value_c
        self.stored_value_d = stored_value_d
        self.assertEqual(size_a, 28)
        self.assertEqual(size_b, 28)
        self.assertEqual(size_c, 305)
        self.assertEqual(size_d, 24)

    def test_store_value(self):
        self.assertEqual(self.stored_value_a, ['a', 42])
        self.assertEqual(self.stored_value_b, ['b', True])
        self.assertEqual(self.stored_value_c, ['c', "S" * 256])
        self.assertEqual(self.stored_value_d, ['d', 3.14])

    def test_restore_value(self):
        self.assertEqual(self.cache_store.restore_value('a', self.stored_value_a), 42)
        self.assertEqual(self.cache_store.restore_value('b', self.stored_value_b), True)
        with self.assertRaises(ValueError):
            self.cache_store.restore_value('e', self.stored_value_b)

    def test_discard_value(self):
        self.cache_store.discard_value('a', self.stored_value_a)
        self.cache_store.discard_value('b', self.stored_value_b)
        self.assertEqual(self.cache_store.restore_value('a', self.stored_value_a), None)
        self.assertEqual(self.cache_store.restore_value('b', self.stored_value_b), None)
        with self.assertRaises(ValueError):
            self.cache_store.discard_value('e', self.stored_value_b)


class FileCacheStoreTest(unittest.TestCase):
    DIR = '__test_file_cache__'

    def setUp(self):
        if os.path.exists(FileCacheStoreTest.DIR):
            shutil.rmtree(FileCacheStoreTest.DIR, ignore_errors=True)

        os.mkdir(FileCacheStoreTest.DIR)
        try:
            self.cache_store = FileCacheStore(FileCacheStoreTest.DIR, ".dat")
            stored_value_a, size_a = self.cache_store.store_value('a', bytes('abc', 'utf8'))
            stored_value_b, size_b = self.cache_store.store_value('b', bytes('def', 'utf8'))
            stored_value_c, size_c = self.cache_store.store_value('c', bytes('ghi', 'utf8'))
            self.stored_value_a = stored_value_a
            self.stored_value_b = stored_value_b
            self.stored_value_c = stored_value_c
            self.assertEqual(size_a, 3)
            self.assertEqual(size_b, 3)
            self.assertEqual(size_c, 3)
        except Exception as e:
            shutil.rmtree(FileCacheStoreTest.DIR, ignore_errors=True)
            raise e

    def tearDown(self):
        shutil.rmtree(FileCacheStoreTest.DIR, ignore_errors=True)

    def test_store_value(self):
        self.assertEqual(self.stored_value_a, os.path.join(FileCacheStoreTest.DIR, 'a.dat'))
        self.assertEqual(self.stored_value_b, os.path.join(FileCacheStoreTest.DIR, 'b.dat'))
        self.assertEqual(self.stored_value_c, os.path.join(FileCacheStoreTest.DIR, 'c.dat'))

    def test_restore_value(self):
        self.assertEqual(self.cache_store.restore_value('a', self.stored_value_a), bytes('abc', 'utf8'))
        self.assertEqual(self.cache_store.restore_value('b', self.stored_value_b), bytes('def', 'utf8'))
        self.assertEqual(self.cache_store.restore_value('c', self.stored_value_c), bytes('ghi', 'utf8'))
        with self.assertRaises(FileNotFoundError):
            self.cache_store.restore_value('e', self.stored_value_b)

    def test_discard_value(self):
        self.cache_store.discard_value('a', self.stored_value_a)
        self.cache_store.discard_value('b', self.stored_value_b)
        self.cache_store.discard_value('c', self.stored_value_c)
        self.cache_store.discard_value('e', self.stored_value_b)
        with self.assertRaises(FileNotFoundError):
            self.cache_store.restore_value('a', self.stored_value_a)
        with self.assertRaises(FileNotFoundError):
            self.cache_store.restore_value('b', self.stored_value_b)
        with self.assertRaises(FileNotFoundError):
            self.cache_store.restore_value('c', self.stored_value_c)


class TracingCacheStore(CacheStore):
    def __init__(self):
        self.trace = ''

    def can_load_from_key(self, key) -> bool:
        self.trace += 'can_load_from_key(%s);' % key
        return key == 'k5'

    def load_from_key(self, key):
        self.trace += 'load_from_key(%s);' % key
        if key == 'k5':
            return 'S/yyyy', 600
        raise ValueError()

    def store_value(self, key, value):
        self.trace += 'store(%s, %s);' % (key, value)
        return 'S/' + value, 100 * len(value)

    def restore_value(self, key, stored_value):
        self.trace += 'restore(%s, %s);' % (key, stored_value)
        return stored_value[2:]

    def discard_value(self, key, stored_value):
        self.trace += 'discard(%s, %s);' % (key, stored_value)


class CacheTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_store_and_restore_and_discard(self):
        cache_store = TracingCacheStore()
        cache = Cache(store=cache_store, capacity=1000)

        self.assertIs(cache.store, cache_store)
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache.max_size, 750)

        cache_store.trace = ''
        cache.put_value('k1', 'x')
        self.assertEqual(cache.get_value('k1'), 'x')
        self.assertEqual(cache.size, 100)
        self.assertEqual(cache_store.trace, 'store(k1, x);restore(k1, S/x);')

        cache_store.trace = ''
        cache.remove_value('k1')
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache_store.trace, 'discard(k1, S/x);')
        cache_store.trace = ''

        cache_store.trace = ''
        cache.put_value('k1', 'x')
        cache.put_value('k1', 'xx')
        self.assertEqual(cache.get_value('k1'), 'xx')
        self.assertEqual(cache.size, 200)
        self.assertEqual(cache_store.trace, 'store(k1, x);discard(k1, S/x);store(k1, xx);restore(k1, S/xx);')

        cache_store.trace = ''
        cache.remove_value('k1')
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache_store.trace, 'discard(k1, S/xx);')

        cache_store.trace = ''
        cache.put_value('k1', 'x')
        cache.put_value('k2', 'xxx')
        cache.put_value('k3', 'xx')
        self.assertEqual(cache.get_value('k1'), 'x')
        self.assertEqual(cache.get_value('k2'), 'xxx')
        self.assertEqual(cache.get_value('k3'), 'xx')
        self.assertEqual(cache.size, 600)
        self.assertEqual(cache_store.trace, 'store(k1, x);store(k2, xxx);store(k3, xx);'
                                            'restore(k1, S/x);restore(k2, S/xxx);restore(k3, S/xx);')

        cache_store.trace = ''
        cache.put_value('k4', 'xxxx')
        self.assertEqual(cache.size, 600)
        self.assertEqual(cache_store.trace, 'store(k4, xxxx);discard(k1, S/x);discard(k2, S/xxx);')

        cache_store.trace = ''
        cache.clear()
        self.assertEqual(cache.size, 0)

    def test_load_from_key(self):
        cache_store = TracingCacheStore()
        cache = Cache(store=cache_store, capacity=1000)

        cache_store.trace = ''
        self.assertEqual(cache.get_value('k1'), None)
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache_store.trace, 'can_load_from_key(k1);')

        cache_store.trace = ''
        self.assertEqual(cache.get_value('k5'), 'yyyy')
        self.assertEqual(cache.size, 600)
        self.assertEqual(cache_store.trace, 'can_load_from_key(k5);load_from_key(k5);restore(k5, S/yyyy);')


class ParseMemSizeTest(unittest.TestCase):
    def test_parse_mem_size(self):
        self.assertEqual(None, parse_mem_size(""))
        self.assertEqual(None, parse_mem_size("0"))
        self.assertEqual(None, parse_mem_size("0M"))
        self.assertEqual(None, parse_mem_size("off"))
        self.assertEqual(None, parse_mem_size("OFF"))
        self.assertEqual(None, parse_mem_size("None"))
        self.assertEqual(None, parse_mem_size("null"))
        self.assertEqual(None, parse_mem_size("False"))
        self.assertEqual(200001, parse_mem_size("200001"))
        self.assertEqual(200001, parse_mem_size("200001B"))
        self.assertEqual(300000, parse_mem_size("300K"))
        self.assertEqual(3000000, parse_mem_size("3M"))
        self.assertEqual(7000000, parse_mem_size("7m"))
        self.assertEqual(2000000000, parse_mem_size("2g"))
        self.assertEqual(2000000000, parse_mem_size("2G"))
        self.assertEqual(1000000000000, parse_mem_size("1T"))

        with self.assertRaises(ValueError) as cm:
            parse_mem_size("7n")
        self.assertEqual("invalid memory size: '7N'", f"{cm.exception}")

        with self.assertRaises(ValueError) as cm:
            parse_mem_size("-2g")
        self.assertEqual("negative memory size: '-2G'", f"{cm.exception}")
