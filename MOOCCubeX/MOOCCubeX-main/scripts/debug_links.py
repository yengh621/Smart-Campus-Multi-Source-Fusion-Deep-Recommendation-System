import json
from collections import defaultdict, Counter

print("=== 1. 检查 concept-course.txt 格式 ===")
with open('relations/concept-course.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        parts = line.strip().split('\t')
        print(f"  行{i}: fields={len(parts)}, 内容: {parts[:3]}")

print("\n=== 2. 检查 course.json 的课程ID格式 ===")
course_num_ids = set()
course_id_to_cid = {}
with open('entities/course.json', 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        cid = rec['id']
        num_part = cid.split('_')[1] if '_' in cid else cid
        course_num_ids.add(num_part)
        course_id_to_cid[num_part] = cid

print(f"  课程总数: {len(course_num_ids)}")
print(f"  课程ID示例: {list(course_num_ids)[:5]}")

print("\n=== 3. user-problem.json 中 problem_id 的数字部分 vs 课程ID ===")
problem_numbers = set()
with open('relations/user-problem.json', 'r', encoding='utf-8') as f:
    count = 0
    for line in f:
        try:
            rec = json.loads(line)
            pid = rec['problem_id']
            if '_' in str(pid):
                num_part = str(pid).split('_')[1]
                problem_numbers.add(num_part)
            count += 1
            if count >= 10000:
                break
        except:
            pass

matched = problem_numbers & course_num_ids
print(f"  problem数字ID数(前10000行): {len(problem_numbers)}")
print(f"  与course.json ID交集: {len(matched)}")
print(f"  交集示例: {list(matched)[:10]}")

print("\n=== 4. user.json 的 course_order 数字 vs course.json ID ===")
with open('entities/user.json', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        rec = json.loads(line)
        uid = rec['id']
        keys = list(rec.keys())
        courses = [str(x) for x in rec.get('course_order', [])]
        matched_courses = set(courses) & course_num_ids
        print(f"  用户{uid}, 字段: {keys}")
        print(f"    选课数: {len(courses)}, 其中匹配 course.json: {len(matched_courses)}")
        print(f"    匹配示例: {list(matched_courses)[:5]}")
        print()

print("=== 5. user.json 总用户数 vs user-problem.json 总用户数 ===")
user_json_users = set()
with open('entities/user.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            rec = json.loads(line)
            user_json_users.add(rec['id'])
        except:
            pass

problem_users = set()
with open('relations/sampled-users.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if parts:
            problem_users.add(parts[0])

overlap = user_json_users & problem_users
print(f"  user.json 用户数: {len(user_json_users)}")
print(f"  sampled-users 用户数: {len(problem_users)}")
print(f"  两者交集: {len(overlap)}")
