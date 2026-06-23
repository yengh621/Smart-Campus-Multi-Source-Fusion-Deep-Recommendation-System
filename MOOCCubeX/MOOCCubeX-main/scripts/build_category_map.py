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

    # 经济学 / 管理学
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
    '农业资源与环境': '机械',

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

CATEGORY_ORDER = ['计算机', '经管', '机械', '艺术', '理学', '医学', '人文', '建筑', '食品农业', '军事', '环境', '其他']


def classify_course(field_list):
    cat_counts = Counter()
    for f in field_list:
        cat = FIELD_TO_CATEGORY.get(f, '其他')
        cat_counts[cat] += 1
    return cat_counts.most_common(1)[0][0] if cat_counts else '其他'


def build_course_mapping():
    print("[1/3] 读取 course.json，建立 course_id → 学科大类映射 ...")
    course_to_cat = {}
    course_to_fields = {}
    cat_counts = Counter()

    with open('entities/course.json', 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            cid = rec['id']
            field_list = rec['field']
            cat = classify_course(field_list)
            course_to_cat[cid] = cat
            course_to_fields[cid] = field_list
            cat_counts[cat] += 1

    print(f"  课程总数: {len(course_to_cat)}")
    print("  学科大类分布:")
    for cat in CATEGORY_ORDER:
        if cat in cat_counts:
            print(f"    {cat}: {cat_counts[cat]}")

    return course_to_cat, course_to_fields


def build_user_to_courses(course_to_cat):
    print("\n[2/3] 读取 user.json，建立 user_id → 学科大类映射 ...")
    user_to_cats = defaultdict(Counter)
    user_enrolled_count = 0

    with open('entities/user.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get('id') or rec.get('user_id')
            if not uid:
                continue
            # user.json 中可能直接有 course 列表字段
            # 也可能有 enrolled_courses / courses 等
            course_ids = []
            for key in ['courses', 'enrolled_courses', 'course_id', 'course_ids']:
                if key in rec and isinstance(rec[key], list):
                    course_ids = rec[key]
                    break
            # 如果是单一字段，尝试从 resource 或其他字段推导
            if not course_ids and 'resource' in rec:
                for res in rec['resource']:
                    if isinstance(res, dict) and 'resource_id' in res:
                        rid = res['resource_id']
                        if rid.startswith('C_'):
                            course_ids.append(rid)

            # 用户可能也在"选课"字段中记录
            # MOOCCubeX 的 user.json 结构可能是: id, name, course_list
            # 先打印几个样本确认结构
            if user_enrolled_count < 3:
                user_enrolled_count += 1
                sample_keys = list(rec.keys())
                print(f"  样本用户 {uid}: keys={sample_keys}")
                if course_ids:
                    print(f"    选课数: {len(course_ids)}, 前5门: {course_ids[:5]}")

            for cid in course_ids:
                if cid in course_to_cat:
                    user_to_cats[uid][course_to_cat[cid]] += 1

    print(f"\n  有学科标签的用户数: {len(user_to_cats)}")
    return user_to_cats


def analyze_and_sample(user_to_cats, sampled_users_path, target=4000):
    print(f"\n[3/3] 从 sampled-users.txt 中读取用户，分析其学科分布 ...")

    sampled_users = set()
    with open(sampled_users_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            sampled_users.add(parts[0])

    print(f"  已抽样用户数: {len(sampled_users)}")

    # 为每个用户确定主学科
    user_main_cat = {}
    cat_user_counts = Counter()
    for uid in sampled_users:
        if uid in user_to_cats:
            main_cat = user_to_cats[uid].most_common(1)[0][0]
        else:
            # 通过 user-problem 中 problem → concept → course 的映射来推断
            main_cat = '未标注'
        user_main_cat[uid] = main_cat
        cat_user_counts[main_cat] += 1

    print("\n  抽样用户的学科分布:")
    for cat in CATEGORY_ORDER + ['未标注']:
        if cat in cat_user_counts:
            print(f"    {cat}: {cat_user_counts[cat]}")

    # 如果学科分布差距大，尝试按学科均衡重采样
    print(f"\n  未标注用户数: {cat_user_counts.get('未标注', 0)}")
    print("  (未标注用户可通过 problem → concept → course 链路推断学科)")

    return user_main_cat


def write_user_category_map(user_main_cat, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for uid, cat in user_main_cat.items():
            f.write(f"{uid}\t{cat}\n")
    print(f"\n已写出: {output_path}")


if __name__ == '__main__':
    course_to_cat, course_to_fields = build_course_mapping()

    # 持久化 course → category 映射
    with open('relations/course-category.txt', 'w', encoding='utf-8') as f:
        for cid, cat in course_to_cat.items():
            f.write(f"{cid}\t{cat}\t{','.join(course_to_fields[cid])}\n")
    print("  → 已写出 relations/course-category.txt")

    user_to_cats = build_user_to_courses(course_to_cat)

    sampled_cat = analyze_and_sample(
        user_to_cats,
        'relations/sampled-users.txt',
        target=4000
    )

    write_user_category_map(sampled_cat, 'relations/sampled-users-category.txt')
    print("\n完成！")
