import json
from collections import defaultdict
import sys

def count_users(input_path, min_records=20, sample_size=4000):
    print(f"[1/3] 扫描 {input_path} 统计每个用户的答题记录数 ...")
    user_counts = defaultdict(int)
    user_times = defaultdict(list)
    total_lines = 0
    bad_lines = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                uid = rec['user_id']
            except (json.JSONDecodeError, KeyError):
                bad_lines += 1
                continue
            user_counts[uid] += 1
            user_times[uid].append(rec.get('submit_time', ''))

    print(f"  总答题记录: {total_lines}")
    print(f"  总用户数: {len(user_counts)}")
    print(f"  解析失败行数: {bad_lines}")

    active_users = {uid: cnt for uid, cnt in user_counts.items() if cnt >= min_records}
    print(f"  >= {min_records} 条的活跃用户: {len(active_users)} 人")

    buckets = {'1-19': 0, '20-49': 0, '50-99': 0, '100-199': 0, '200+': 0}
    for uid, cnt in user_counts.items():
        if cnt < 20: buckets['1-19'] += 1
        elif cnt < 50: buckets['20-49'] += 1
        elif cnt < 100: buckets['50-99'] += 1
        elif cnt < 200: buckets['100-199'] += 1
        else: buckets['200+'] += 1
    print("  答题数分布:")
    for k, v in buckets.items():
        print(f"    {k} 条: {v} 人")

    return active_users, user_counts, user_times


def check_time_order(user_times, active_users):
    print(f"\n[2/3] 检查活跃用户的答题时间是否有序 ...")
    disordered = 0
    for uid in list(active_users.keys())[:100]:
        times = user_times[uid]
        if times != sorted(times):
            disordered += 1
    print(f"  抽样 100 个活跃用户中，时间乱序的: {disordered} 个")
    print(f"  结论：训练 AKT 前需要按 submit_time 对每个用户的记录重新排序")


def sample_users(user_counts, min_records=20, target=4000):
    print(f"\n[3/3] 从 >= {min_records} 条的用户中抽 {target} 人 (均衡答题数) ...")
    active = [(uid, cnt) for uid, cnt in user_counts.items() if cnt >= min_records]

    # 按答题数分层抽样
    low = [uid for uid, cnt in active if 20 <= cnt < 50]
    mid = [uid for uid, cnt in active if 50 <= cnt < 100]
    high = [uid for uid, cnt in active if 100 <= cnt < 200]
    very_high = [uid for uid, cnt in active if cnt >= 200]

    print(f"  分层: 20-49条={len(low)}, 50-99条={len(mid)}, 100-199条={len(high)}, 200+={len(very_high)}")

    import random
    random.seed(42)
    n_each = target // 4
    extra = target % 4

    sampled = (random.sample(low, min(n_each, len(low))) +
               random.sample(mid, min(n_each, len(mid))) +
               random.sample(high, min(n_each, len(high))) +
               random.sample(very_high, min(n_each + extra, len(very_high))))

    if len(sampled) > target:
        sampled = random.sample(sampled, target)

    print(f"  实际抽样: {len(sampled)} 人")

    return set(sampled)


def write_filtered_data(input_path, sampled_users, output_path, user_times):
    print(f"\n[4/4] 写出 {len(sampled_users)} 名用户的完整时序数据到 {output_path} ...")
    sampled_records = defaultdict(list)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                uid = rec['user_id']
            except (json.JSONDecodeError, KeyError):
                continue
            if uid in sampled_users:
                sampled_records[uid].append(rec)

    total_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for uid in sorted(sampled_users):
            records = sorted(sampled_records[uid], key=lambda r: r.get('submit_time', ''))
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                total_written += 1

    print(f"  共写出 {total_written} 条答题记录 (已按时间排序)")
    print(f"  输出文件: {output_path}")


def write_user_list(sampled_users, output_path, user_counts):
    with open(output_path, 'w', encoding='utf-8') as f:
        for uid in sorted(sampled_users):
            f.write(f"{uid}\t{user_counts[uid]}\n")
    print(f"  用户列表: {output_path}")


if __name__ == '__main__':
    input_file = 'relations/user-problem.json'
    filtered_file = 'relations/user-problem-filtered.json'
    user_list_file = 'relations/sampled-users.txt'

    min_records = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    print("=" * 60)
    print(f"筛选条件: 答题 >= {min_records} 条, 抽样 {target} 人")
    print("=" * 60)

    active_users, user_counts, user_times = count_users(input_file, min_records, target)
    check_time_order(user_times, active_users)

    if len(active_users) < target:
        print(f"\n警告: 活跃用户只有 {len(active_users)} 人，小于目标 {target} 人")
        target = len(active_users)

    sampled = sample_users(user_counts, min_records, target)

    write_filtered_data(input_file, sampled, filtered_file, user_times)
    write_user_list(sampled, user_list_file, user_counts)

    print("\n完成！")
