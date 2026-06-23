import json
from collections import Counter

fields_counter = Counter()
course_fields = {}
course_names = {}

with open('entities/course.json', 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        cid = rec['id']
        field_list = rec['field']
        course_fields[cid] = field_list
        course_names[cid] = rec['name']
        for field in field_list:
            fields_counter[field] += 1

print('=== 全部 field 值及其出现次数 (前40) ===')
for field, cnt in fields_counter.most_common(40):
    print(f'  {field}: {cnt}')

print(f'\n课程总数: {len(course_fields)}')
print(f'不同的 field 类别数: {len(fields_counter)}')

# 看 field 的实际字符串格式 (是否是中文/英文)
print('\n=== 前 20 个 field 字符串样本 ===')
for field in list(fields_counter.keys())[:20]:
    print(f'  {repr(field)}')
