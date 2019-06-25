import datetime
import hashlib
import os.path
import os.path
from typing import Dict, Optional, Any, List


def list_bucket(bucket_path: str,
                delimiter: str = None,
                prefix: str = None,
                max_keys: int = None,
                marker: str = None,
                storage_class: str = None,
                last_modified: str = None) -> Optional[Dict]:
    bucket_path = os.path.abspath(os.path.normpath(bucket_path))
    bucket_name = os.path.basename(bucket_path)
    if not os.path.isdir(bucket_path):
        # Get object
        return None

    max_keys = max_keys or 1000
    storage_class = storage_class or 'STANDARD'

    contents_list = []
    is_truncated = False
    next_marker = None
    marker_seen = marker is None
    common_prefixes_list = []
    common_prefixes_set = set()

    for root, dirs, files in os.walk(bucket_path):

        key_prefix = root[len(bucket_path):].replace('\\', '/') + '/'
        if key_prefix[0] == '/':
            key_prefix = key_prefix[1:]

        if key_prefix == '':
            continue

        for file in [''] + files:
            key = key_prefix + file

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

            path = os.path.join(root, file) if file else root
            stat = os.stat(path)
            item = dict(Key=key,
                        Size=0 if key[-1] == '/' else stat.st_size,
                        LastModified=last_modified or str(datetime.datetime.fromtimestamp(stat.st_mtime)),
                        ETag='"' + hashlib.md5(bytes(path, encoding='utf-8')).hexdigest() + '"',
                        StorageClass=storage_class)
            contents_list.append(item)

    list_bucket_result = dict(Name=bucket_name,
                              Prefix=prefix,
                              Delimiter=delimiter,
                              MaxKeys=max_keys,
                              IsTruncated=is_truncated,
                              Marker=marker,
                              NextMarker=next_marker)
    if contents_list:
        list_bucket_result.update(Contents=contents_list)
    if common_prefixes_list:
        list_bucket_result.update(CommonPrefixes=common_prefixes_list)
    return list_bucket_result


def list_bucket_result_to_xml(list_bucket_result):
    return dict_to_xml('ListBucketResult',
                       list_bucket_result,
                       root_element_attrs=dict(xmlns="http://obs.otc.t-systems.com/doc/2016-01-01/"))


def dict_to_xml(root_element_name: str, content_dict: Dict, root_element_attrs:Dict = None) -> str:
    lines = []
    _value_to_xml(lines, root_element_name, content_dict, element_attrs=root_element_attrs)
    return '\n'.join(lines)


def _value_to_xml(lines: List[str], element_name: str, element_value: Any, element_attrs: Dict = None, indent=0):
    attrs = ''
    if element_attrs:
        for attr_name, attr_value in element_attrs.items():
            attrs += f' {attr_name}=\"{attr_value}\"'
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
        lines.append(f'{indent * "  "}<{element_name}{attrs}>{element_value}</{element_name}>')
