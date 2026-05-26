<!-- markdownlint-disable MD041 -->
当传递 JSON CLI 参数时，以下两组参数是等效的：

- `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
- `--json-arg.key1 value1 --json-arg.key2.key3 value2`

此外，列表元素可以使用 `+` 逐个传递：

- `--json-arg '{"key4": ["value3", "value4", "value5"]}'`
- `--json-arg.key4+ value3 --json-arg.key4+='value4,value5'`
