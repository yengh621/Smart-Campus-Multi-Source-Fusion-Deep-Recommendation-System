#! /bin/bash

BASE_URL="https://lfs.aminer.cn/misc/moocdata/data/mooccube2"
MAX_RETRIES=10
RETRY_SLEEP=5

[[ -d entities ]] || mkdir entities
[[ -d relations ]] || mkdir relations
[[ -d prerequisites ]] || mkdir prerequisites

download_file() {
    local filepath="$1"
    local url="${BASE_URL}/${filepath}"
    local attempt=1
    echo "Downloading ${filepath} ..."
    while [ $attempt -le $MAX_RETRIES ]; do
        if curl -fsSL --connect-timeout 30 --max-time 600 --retry 3 --retry-delay 3 --create-dirs "$url" -o "$filepath" -C -; then
            echo "  ✓ ${filepath} 下载完成"
            return 0
        else
            echo "  ✗ 第 ${attempt} 次失败，${RETRY_SLEEP}秒后重试..."
            attempt=$((attempt + 1))
            sleep $RETRY_SLEEP
        fi
    done
    echo "  ✗✗✗ ${filepath} 下载失败，已达最大重试次数"
    return 1
}

download_file "entities/course.json"
download_file "entities/concept.json"
download_file "entities/user.json"
download_file "entities/problem.json"
download_file "relations/concept-course.txt"
download_file "relations/course-field.json"
download_file "relations/concept-problem.txt"
download_file "prerequisites/cs.json"
download_file "prerequisites/math.json"

echo ""
echo "=== 直接下载文件完成 ==="
echo ""
echo "提示："
echo "  user_course.json 的信息已包含在 entities/user.json 中"
echo "  course-field.json 为课程的人工标注学科分类"
echo "  concept-problem.txt 为题目-知识点关联"
