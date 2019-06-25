import datetime
import hashlib
import os.path
import os.path
from typing import Dict, Any, List, Tuple, Iterator


def list_bucket_v2(bucket_entries: Dict[str, str],
                   name: str = None,
                   delimiter: str = None,
                   prefix: str = None,
                   max_keys: int = None,
                   start_after: str = None,
                   continuation_token: str = None,
                   storage_class: str = None,
                   last_modified: str = None) -> Dict:
    name = name or 's3bucket'
    max_keys = max_keys or 1000
    start_after = None if continuation_token else start_after
    storage_class = storage_class or 'STANDARD'

    contents_list = []
    is_truncated = False
    next_continuation_token = None
    continuation_token_seen = continuation_token is None
    start_key_seen = start_after is None
    common_prefixes_list = []
    common_prefixes_set = set()

    for key, path in list_bucket_keys(bucket_entries):

        if len(contents_list) == max_keys:
            is_truncated = True
            next_continuation_token = key
            break

        if key == start_after:
            start_key_seen = True
            continue

        # Note, for time being we use the key as token,
        # but actually any token is more useful than key because it allows continuing
        # the walk through keys at given app-defined token.
        # This is the major change from S3 API v1 to v2.
        if key == continuation_token:
            continuation_token_seen = True

        if not continuation_token_seen or not start_key_seen:
            continue

        if prefix and not key.startswith(prefix):
            continue

        if delimiter:
            index = key.find(delimiter, len(prefix) if prefix else 0)
            if index >= 0:
                key = key[:index + len(delimiter)]
                if key not in common_prefixes_set:
                    common_prefixes_set.add(key)
                    common_prefixes_list.append(key)
                continue

        stat = os.stat(path)
        item = dict(Key=key,
                    Size=0 if key[-1] == '/' else stat.st_size,
                    LastModified=last_modified or mtime_to_str(stat.st_mtime),
                    ETag='"' + path_to_md5(path) + '"',
                    StorageClass=storage_class)
        contents_list.append(item)

    list_bucket_result = dict(Name=name,
                              Prefix=prefix,
                              StartAfter=start_after,
                              MaxKeys=max_keys,
                              Delimiter=delimiter,
                              IsTruncated=is_truncated,
                              ContinuationToken=continuation_token)
    if is_truncated:
        list_bucket_result.update(NextContinuationToken=next_continuation_token)
    if contents_list:
        list_bucket_result.update(Contents=contents_list)
    if common_prefixes_list:
        list_bucket_result.update(CommonPrefixes=[dict(Prefix=prefix) for prefix in common_prefixes_list])
    return list_bucket_result


def list_bucket_v1(bucket_entries: Dict[str, str],
                   name: str = None,
                   delimiter: str = None,
                   prefix: str = None,
                   max_keys: int = None,
                   marker: str = None,
                   storage_class: str = None,
                   last_modified: str = None) -> Dict:
    name = name or 's3bucket'
    max_keys = max_keys or 1000
    storage_class = storage_class or 'STANDARD'

    contents_list = []
    is_truncated = False
    next_marker = None
    marker_seen = marker is None
    common_prefixes_list = []
    common_prefixes_set = set()

    for key, path in list_bucket_keys(bucket_entries):

        if len(contents_list) == max_keys:
            is_truncated = True
            next_marker = key
            break

        if key == marker:
            marker_seen = True

        if not marker_seen:
            continue

        if prefix and not key.startswith(prefix):
            continue

        if delimiter:
            index = key.find(delimiter, len(prefix) if prefix else 0)
            if index >= 0:
                key = key[:index + len(delimiter)]
                if key not in common_prefixes_set:
                    common_prefixes_set.add(key)
                    common_prefixes_list.append(key)
                continue

        stat = os.stat(path)
        item = dict(Key=key,
                    Size=0 if key[-1] == '/' else stat.st_size,
                    LastModified=last_modified or mtime_to_str(stat.st_mtime),
                    ETag='"' + path_to_md5(path) + '"',
                    StorageClass=storage_class)
        contents_list.append(item)

    list_bucket_result = dict(Name=name,
                              Prefix=prefix,
                              Marker=marker,
                              MaxKeys=max_keys,
                              Delimiter=delimiter,
                              IsTruncated=is_truncated)
    if is_truncated:
        list_bucket_result.update(NextMarker=next_marker)
    if contents_list:
        list_bucket_result.update(Contents=contents_list)
    if common_prefixes_list:
        list_bucket_result.update(CommonPrefixes=[dict(Prefix=prefix) for prefix in common_prefixes_list])
    return list_bucket_result


def list_bucket_keys(bucket_entries: Dict[str, str]) -> Iterator[Tuple[str, str]]:
    bucket_entry_keys = sorted(list(bucket_entries.keys()))

    for bucket_entry_key in bucket_entry_keys:

        bucket_entry_path = bucket_entries[bucket_entry_key]
        bucket_entry_path = os.path.abspath(os.path.normpath(bucket_entry_path))
        if not os.path.isdir(bucket_entry_path):
            raise ValueError(f'Value for key {bucket_entry_key!r} is not a directory: {bucket_entry_path}')

        yield bucket_entry_key + '/', bucket_entry_path

        for root, dirs, files in os.walk(bucket_entry_path):

            dirs.sort()
            files.sort()

            sub_key = root[len(bucket_entry_path):].replace('\\', '/') + '/'
            if sub_key[0] == '/':
                sub_key = sub_key[1:]

            if sub_key:
                yield bucket_entry_key + '/' + sub_key, root

            for file in files:
                yield bucket_entry_key + '/' + sub_key + file, os.path.join(root, file)


def list_bucket_result_to_xml(list_bucket_result):
    return dict_to_xml('ListBucketResult',
                       list_bucket_result,
                       root_element_attrs=dict(xmlns="http://s3.amazonaws.com/doc/2006-03-01/"))


def dict_to_xml(root_element_name: str, content_dict: Dict, root_element_attrs: Dict = None) -> str:
    lines = []
    _value_to_xml(lines, root_element_name, content_dict, element_attrs=root_element_attrs)
    return '\n'.join(lines)


def _value_to_xml(lines: List[str],
                  element_name: str,
                  element_value: Any,
                  element_attrs: Dict = None,
                  indent: int = 0):
    attrs = ''
    if element_attrs:
        for attr_name, attr_value in element_attrs.items():
            attrs += f' {attr_name}=\"{_value_to_text(attr_value)}\"'
    if element_value is None:
        lines.append(f'{indent * "  "}<{element_name}{attrs}/>')
    elif isinstance(element_value, dict):
        lines.append(f'{indent * "  "}<{element_name}{attrs}>')
        for sub_element_name, sub_element_value in element_value.items():
            _value_to_xml(lines, sub_element_name, sub_element_value, indent=indent + 1)
        lines.append(f'{indent * "  "}</{element_name}>')
    elif isinstance(element_value, list):
        for item in element_value:
            _value_to_xml(lines, element_name, item, indent=indent)
    else:
        lines.append(f'{indent * "  "}<{element_name}{attrs}>{_value_to_text(element_value)}</{element_name}>')


def _value_to_text(value) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def mtime_to_str(mtime) -> str:
    return str(datetime.datetime.fromtimestamp(mtime))


def path_to_md5(path) -> str:
    return hashlib.md5(bytes(path, encoding='utf-8')).hexdigest()
