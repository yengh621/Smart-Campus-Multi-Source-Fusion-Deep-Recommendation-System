import json
from collections import defaultdict, Counter
import random

FIELD_TO_CATEGORY = {
    # 计算机 / 电子信息
    '计算机科学与技术': '计算机',
    '软件工程': '计算机',
    '电子科学与技术': '计算机',
    '信息与通信工程': '计算机',
    '控制科学与工程': '计算机',
    '系统科学': '计算机',
    '光学工程': '计算机',
    '网络空间安全': '计算机',
    '软件工程技术': '计算机',
    '电子与通信工程': '计算机',
    '计算机技术': '计算机',
    # 经济 / 管理
    '应用经济学': '经管',
    '理论经济学': '经管',
    '工商管理': '经管',
    '管理科学与工程': '经管',
    '公共管理': '经管',
    '农林经济管理': '经管',
    '图书情报与档案管理': '经管',
    '情报与档案管理': '经管',
    # 机械 / 工程
    '机械工程': '机械',
    '材料科学与工程': '机械',
    '动力工程及工程热物理': '机械',
    '电气工程': '机械',
    '土木工程': '机械',
    '水利工程': '机械',
    '化学工程与技术': '机械',
    '化学工程': '机械',
    '航空宇航科学与技术': '机械',
    '兵器科学与技术': '机械',
    '农业工程': '机械',
    '仪器科学与技术': '机械',
    '冶金工程': '机械',
    '矿业工程': '机械',
    '石油与天然气工程': '机械',
    '纺织科学与工程': '机械',
    '轻工技术与工程': '机械',
    '交通运输工程': '机械',
    '船舶与海洋工程': '机械',
    '航空工程': '机械',
    '兵器工程': '机械',
    '车辆工程': '机械',
    '工业设计工程': '机械',
    '机械': '机械',
    '测绘科学与技术': '机械',
    '地质资源与地质工程': '机械',
    '核科学与技术': '机械',
    # 艺术 / 设计
    '艺术学': '艺术',
    '美术学': '艺术',
    '设计学': '艺术',
    '音乐与舞蹈学': '艺术',
    '戏剧与影视学': '艺术',
    '新闻传播学': '艺术',
    # 理学
    '数学': '理学',
    '物理学': '理学',
    '化学': '理学',
    '生物学': '理学',
    '地理学': '理学',
    '大气科学': '理学',
    '海洋科学': '理学',
    '地球物理学': '理学',
    '地质学': '理学',
    '天文学': '理学',
    '生态学': '理学',
    '统计学': '理学',
    # 医学 / 药学
    '临床医学': '医学',
    '基础医学': '医学',
    '公共卫生与预防医学': '医学',
    '药学': '医学',
    '中药学': '医学',
    '护理学': '医学',
    '口腔医学': '医学',
    '中医学': '医学',
    '中西医结合': '医学',
    # 人文社科
    '哲学': '人文',
    '政治学': '人文',
    '社会学': '人文',
    '民族学': '人文',
    '马克思主义理论': '人文',
    '教育学': '人文',
    '心理学': '人文',
    '体育学': '人文',
    '中国语言文学': '人文',
    '外国语言文学': '人文',
    '历史学': '人文',
    '世界史': '人文',
    '考古学': '人文',
    '中国史': '人文',
    '法学': '人文',
    # 建筑 / 规划
    '建筑学': '建筑',
    '城乡规划学': '建筑',
    '风景园林学': '建筑',
    # 食品 / 农业
    '食品科学与工程': '食品农业',
    '食品工程': '食品农业',
    '园艺学': '食品农业',
    '畜牧学': '食品农业',
    '兽医学': '食品农业',
    '林学': '食品农业',
    '水产': '食品农业',
    # 军事
    '军队政治工作学': '军事',
    '军事思想及军事历史': '军事',
    '军事战略学': '军事',
    '战役学': '军事',
    '战术学': '军事',
    '军队指挥学': '军事',
    '军制学': '军事',
    '军事后勤学与军事装备学': '军事',
    # 环境 / 资源
    '环境科学与工程': '环境',
    '环境工程': '环境',
    '安全科学与工程': '环境',
    # 其他 / 未分类
    '科学技术史': '其他',
}


def classify_course(field_list):
    cat_counts = Counter()
    for f in field_list:
        cat = FIELD_TO_CATEGORY.get(f, '其他')
        cat_counts[cat] += 1
    return cat_counts.most_common(1)[0][0] if cat_counts else '其他'


print("=" * 70)
print("步骤 1: 读取 course.json → 建立课程ID → 学科大类映射")
print("=" * 70)
course_num_to_cat = {}       # '584313' -> '人文'
course_cid_to_cat = {}        # 'C_584313' -> '人文'
course_cid_to_fields = {}     # 'C_584313' -> ['历史学', ...]
cat_course_count = Counter()

with open('entities/course.json', 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        cid = rec['id']
        num_part = cid.split('_')[1] if '_' in cid else cid
        field_list = rec['field']
        cat = classify_course(field_list)
        course_num_to_cat[num_part] = cat
        course_cid_to_cat[cid] = cat
        course_cid_to_fields[cid] = field_list
        cat_course_count[cat] += 1

print(f"  课程总数: {len(course_cid_to_cat)}")
print(f"  学科大类分布:")
for cat, cnt in cat_course_count.most_common():
    print(f"    {cat}: {cnt}")

print("\n" + "=" * 70)
print("步骤 2: 读取 user-problem.json → 统计每个用户的答题数")
print("=" * 70)
user_problem_count = defaultdict(int)
problem_bad_lines = 0

with open('relations/user-problem.json', 'r', encoding='utf-8') as f:
    for line_no, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            uid = rec['user_id']
            user_problem_count[uid] += 1
        except (json.JSONDecodeError, KeyError):
            problem_bad_lines += 1
            continue

total_users = len(user_problem_count)
active_users = {uid: cnt for uid, cnt in user_problem_count.items() if cnt >= 20}
print(f"  有答题记录的用户: {total_users}")
print(f"  解析失败行: {problem_bad_lines}")
print(f"  >=20 题的活跃用户: {len(active_users)}")

print("\n" + "=" * 70)
print("步骤 3: 读取 user.json → 给用户打学科标签 (基于 course_order)")
print("=" * 70)
user_to_cat = {}      # uid -> '计算机'
user_cat_counter = {} # uid -> Counter of categories (for debug)
users_with_course = 0
users_with_labeled_course = 0

with open('entities/user.json', 'r', encoding='utf-8') as f:
    for line_no, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            uid = rec['id']
        except:
            continue
        courses = rec.get('course_order', [])
        if not courses:
            continue
        users_with_course += 1
        cat_counts = Counter()
        for cid_num in courses:
            cid_str = str(cid_num)
            if cid_str in course_num_to_cat:
                cat_counts[course_num_to_cat[cid_str]] += 1
        if cat_counts:
            users_with_labeled_course += 1
            # 取选课最多的学科作为主学科
            main_cat = cat_counts.most_common(1)[0][0]
            user_to_cat[uid] = main_cat
            user_cat_counter[uid] = cat_counts

print(f"  user.json 中有选课记录的用户: {users_with_course}")
print(f"  其中能打上学科标签的用户: {users_with_labeled_course}")

print("\n" + "=" * 70)
print("步骤 4: 交集 → 有学科标签 + 有答题记录 (≥20)")
print("=" * 70)
# 活跃用户 ∩ 有学科标签的用户
qualified = []
for uid, cnt in active_users.items():
    if uid in user_to_cat:
        qualified.append((uid, cnt, user_to_cat[uid]))

print(f"  满足条件的用户数: {len(qualified)}")

by_cat = defaultdict(list)
for uid, cnt, cat in qualified:
    by_cat[cat].append((uid, cnt))

print(f"  按学科大类分布:")
total_target = 4000
for cat, users in sorted(by_cat.items(), key=lambda x: -len(x[1])):
    print(f"    {cat}: {len(users)} 人 (平均答题 {sum(c for _,c in users)/len(users):.0f} 题)")

print("\n" + "=" * 70)
print(f"步骤 5: 按学科均衡抽样 {total_target} 人")
print("=" * 70)
# 按每个学科人数的比例抽样，保证小类也有代表
random.seed(42)
# 合并小类 (人数 < 50 的并入 "其他")
major_cats = {cat: users for cat, users in by_cat.items() if len(users) >= 50}
minor_users = []
for cat, users in by_cat.items():
    if len(users) < 50:
        minor_users.extend(users)
if minor_users:
    major_cats['其他'] = major_cats.get('其他', []) + minor_users

total_qualified = sum(len(v) for v in major_cats.values())
print(f"  总合格用户(含小类合并后): {total_qualified}")

# 均衡抽样: 每个学科抽取 n = total_target / num_cats, 但不超过该学科的可用人数
num_cats = len(major_cats)
per_cat = total_target // num_cats
sampled = []
sampled_distribution = {}

for cat, users in major_cats.items():
    n_sample = min(per_cat, len(users))
    # 按答题数降序排序后抽样，优先保留时序完整的用户
    sorted_users = sorted(users, key=lambda x: -x[1])
    # 从排名前 50% 中随机抽样，既保证活跃又避免偏置
    top_half = sorted_users[:max(len(sorted_users) // 2, n_sample)]
    picked = random.sample(top_half, min(n_sample, len(top_half)))
    sampled.extend([(uid, cnt, cat) for uid, cnt in picked])
    sampled_distribution[cat] = len(picked)

# 如果抽样不够，从最大的学科中补齐
while len(sampled) < total_target:
    largest_cat = max(major_cats.items(), key=lambda x: len(x[1]))
    remaining = [(uid, cnt) for uid, cnt in largest_cat[1] if not any(s[0] == uid for s in sampled)]
    if not remaining:
        break
    needed = total_target - len(sampled)
    picked = random.sample(remaining, min(needed, len(remaining)))
    sampled.extend([(uid, cnt, largest_cat[0]) for uid, cnt in picked])

print(f"  实际抽样: {len(sampled)} 人")
print(f"  抽样分布:")
dist_count = Counter(cat for _, _, cat in sampled)
for cat, cnt in sorted(dist_count.items(), key=lambda x: -x[1]):
    print(f"    {cat}: {cnt}")

print("\n" + "=" * 70)
print("步骤 6: 写出结果文件")
print("=" * 70)

sampled_user_ids = set(uid for uid, _, _ in sampled)

# 6a: 写出 用户 ID + 学科标签 列表
with open('relations/final-users.csv', 'w', encoding='utf-8') as f:
    f.write("user_id,problem_count,category\n")
    for uid, cnt, cat in sorted(sampled, key=lambda x: x[2]):
        f.write(f"{uid},{cnt},{cat}\n")
print(f"  OK relations/final-users.csv ({len(sampled)} 用户)")

# 6b: 写出筛选后的答题时序数据 (按 submit_time 排序)
user_records = defaultdict(list)
with open('relations/user-problem.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            uid = rec['user_id']
        except (json.JSONDecodeError, KeyError):
            continue
        if uid in sampled_user_ids:
            user_records[uid].append(rec)

total_written = 0
with open('relations/user-problem-final.json', 'w', encoding='utf-8') as f:
    for uid in sorted(sampled_user_ids):
        records = sorted(user_records[uid], key=lambda r: r.get('submit_time', ''))
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            total_written += 1

print(f"  OK relations/user-problem-final.json ({total_written} 条答题记录, 已排序)")

# 6c: 写出课程 → 学科映射
with open('relations/course-category.csv', 'w', encoding='utf-8') as f:
    f.write("course_id,category,fields\n")
    for cid, fields in course_cid_to_fields.items():
        cat = course_cid_to_cat[cid]
        f.write(f"{cid},{cat},{'|'.join(fields)}\n")
print(f"  OK relations/course-category.csv ({len(course_cid_to_cat)} 门课)")

print("\n" + "=" * 70)
print("完成! 输出文件汇总:")
print("=" * 70)
print("  1. relations/final-users.csv           - 4000用户的学科标签")
print("  2. relations/user-problem-final.json   - 这4000用户的全部答题记录")
print("  3. relations/course-category.csv       - 3781门课的学科标签")
