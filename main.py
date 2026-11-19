"""
Mini NPU Simulator
------------------
MAC(Multiply-Accumulate) 연산으로 입력 패턴이 어떤 필터(Cross / X)와
더 유사한지 판정하는 콘솔 시뮬레이터.

- 외부 라이브러리 없이 표준 라이브러리(json, time, re, math, argparse)만 사용
- Python 3.8 이상

실행:
    python main.py                      # data.json (main.py와 같은 폴더) 사용
    python main.py --data other.json    # 다른 JSON 파일로 모드 2 실행
"""

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata

# ---------------------------------------------------------------------------
# 설정 상수
# ---------------------------------------------------------------------------
EPSILON = 1e-9          # 점수 비교 허용오차: |a - b| < EPSILON 이면 동점
REPEAT = 10             # 성능 측정 반복 횟수 (크기별 평균)
PERF_SIZES = [3, 5, 13, 25]
BONUS_SIZES = [3, 5, 13, 25, 51, 101]

LABEL_CROSS = "Cross"
LABEL_X = "X"
STANDARD_LABELS = (LABEL_CROSS, LABEL_X)

# 외부 표기 -> 표준 라벨. 새 라벨은 이 표에만 추가하면 된다.
LABEL_ALIASES = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

FILTER_KEY_RE = re.compile(r"^size_(\d+)$")
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")


class InputClosed(Exception):
    """표준 입력이 닫혔을 때(EOF) 사용."""


# ---------------------------------------------------------------------------
# 1. 데이터 구조: n×n 2차원 배열 (list of list)
# ---------------------------------------------------------------------------
def make_matrix(n, fill=0.0):
    """n×n 크기의 2차원 배열을 만든다."""
    return [[fill for _ in range(n)] for _ in range(n)]


def get_value(matrix, row, col):
    """특정 위치의 값을 읽는다."""
    return matrix[row][col]


def set_value(matrix, row, col, value):
    """특정 위치에 값을 저장한다."""
    matrix[row][col] = value


def matrix_size(matrix):
    return len(matrix)


def to_matrix(raw, n):
    """
    JSON에서 읽은 값(raw)을 검증해 n×n float 행렬로 변환한다.
    문제가 있으면 ValueError(사유)를 던진다.
    """
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("2차원 배열이 아닙니다")
    if len(raw) != n:
        raise ValueError("행 수 불일치: 기대 {0}, 실제 {1}".format(n, len(raw)))
    result = make_matrix(n)
    for r, row in enumerate(raw):
        if not isinstance(row, list):
            raise ValueError("{0}행이 배열이 아닙니다".format(r + 1))
        if len(row) != n:
            raise ValueError("{0}행 열 수 불일치: 기대 {1}, 실제 {2}".format(r + 1, n, len(row)))
        for c, v in enumerate(row):
            # bool은 int의 하위 타입이므로 명시적으로 제외
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("({0},{1}) 위치 값이 숫자가 아닙니다: {2!r}".format(r + 1, c + 1, v))
            set_value(result, r, c, float(v))
    return result


# ---------------------------------------------------------------------------
# 2. 라벨 정규화
# ---------------------------------------------------------------------------
def normalize_label(raw):
    """
    외부 표기('+', 'x', 'cross', 'Cross', ...)를 표준 라벨(Cross / X)로 변환한다.
    알 수 없는 라벨이면 None을 반환한다.
    """
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


# ---------------------------------------------------------------------------
# 3. MAC 연산 (반복문 직접 구현)
# ---------------------------------------------------------------------------
def mac(pattern, filt):
    """
    위치별 곱셈 후 누적 합산.
    입력 → (i, j) 위치별 곱셈 → 누적 합산 → 점수 반환
    """
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


# ---------------------------------------------------------------------------
# 4. 점수 비교 정책 (epsilon)
# ---------------------------------------------------------------------------
def compare_scores(score_a, score_b, eps=EPSILON):
    """
    두 점수를 허용오차 기반으로 비교한다.
    반환값: 'A' (A가 큼), 'B' (B가 큼), None (동점)
    """
    if abs(score_a - score_b) < eps:
        return None
    return "A" if score_a > score_b else "B"


# ---------------------------------------------------------------------------
# 5. 성능 측정 (연산 구간만)
# ---------------------------------------------------------------------------
def measure(fn, *args, repeat=REPEAT):
    """
    fn(*args)를 repeat번 호출해 1회 평균 시간(ms)을 반환한다.
    입력 준비/출력은 측정 구간 밖에서 끝낸 뒤 호출해야 한다.
    """
    fn(*args)  # 워밍업 1회 (측정 제외)
    elapsed = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        elapsed += time.perf_counter() - start
    return elapsed / repeat * 1000.0


# ---------------------------------------------------------------------------
# 6. 패턴 생성기 (보너스)
# ---------------------------------------------------------------------------
def generate_cross(n, value=1.0):
    """가운데 행과 가운데 열이 채워진 N×N 십자가 패턴."""
    m = make_matrix(n)
    mid = n // 2
    for i in range(n):
        set_value(m, mid, i, value)
        set_value(m, i, mid, value)
    return m


def generate_x(n, value=1.0):
    """두 대각선이 채워진 N×N X 패턴."""
    m = make_matrix(n)
    for i in range(n):
        set_value(m, i, i, value)
        set_value(m, i, n - 1 - i, value)
    return m


# ---------------------------------------------------------------------------
# 7. 1차원 최적화 (보너스)
# ---------------------------------------------------------------------------
def flatten(matrix):
    """N×N 2차원 배열을 길이 N² 1차원 배열로 변환한다."""
    flat = []
    for row in matrix:
        for v in row:
            flat.append(v)
    return flat


def mac_flat(pattern_flat, filter_flat):
    """1차원 배열 기반 MAC: 인덱스 하나로 순차 접근."""
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filter_flat[k]
    return total


# ---------------------------------------------------------------------------
# 콘솔 출력 도우미
# ---------------------------------------------------------------------------
def section(title):
    print()
    print("#" + "-" * 40)
    print("# " + title)
    print("#" + "-" * 40)


def fmt_score(score):
    """부동소수점 오차가 그대로 보이도록 repr로 출력한다."""
    return repr(score)


def print_matrix(matrix, indent="  "):
    for row in matrix:
        print(indent + " ".join("{0:g}".format(v) for v in row))


def display_width(text):
    """한글 등 전각 문자를 2칸으로 계산한 표시 폭."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text, width, align="<"):
    """표시 폭 기준으로 공백을 채운다 (한글 열 정렬용)."""
    text = str(text)
    space = " " * max(0, width - display_width(text))
    return text + space if align == "<" else space + text


def print_table(headers, rows, widths):
    """첫 열은 왼쪽, 나머지는 오른쪽 정렬로 표를 출력한다."""
    def line(cells):
        return "".join(pad(c, w, "<" if i == 0 else ">") for i, (c, w) in enumerate(zip(cells, widths)))
    print(line(headers))
    print("-" * sum(widths))
    for r in rows:
        print(line(r))


def print_perf_table(rows):
    """rows: [(n, avg_ms), ...] -> 크기 / 평균 시간(ms) / 연산 횟수(N²) / 연산당(ns)"""
    table = []
    for n, avg_ms in rows:
        ops = n * n
        table.append(["{0}×{0}".format(n), "{0:.4f}".format(avg_ms), str(ops),
                      "{0:.1f}".format(avg_ms * 1e6 / ops)])
    print_table(["크기(N×N)", "평균 시간(ms)", "연산 횟수(N²)", "연산당(ns)"], table, [10, 16, 16, 14])


def run_perf(sizes, filters_by_size=None):
    """
    크기별 MAC 평균 시간을 측정한다.
    - 패턴: 생성기로 만든 Cross 패턴
    - 필터: data.json 필터가 있으면 그 Cross 필터, 없으면 생성기 필터
    입력 준비는 모두 측정 전에 끝낸다.
    """
    rows = []
    for n in sizes:
        pattern = generate_cross(n)
        filt = None
        if filters_by_size and n in filters_by_size:
            filt = filters_by_size[n].get(LABEL_CROSS)
        if filt is None:
            filt = generate_cross(n)
        rows.append((n, measure(mac, pattern, filt)))
    return rows


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ---------------------------------------------------------------------------
def read_line(prompt=""):
    try:
        return input(prompt)
    except EOFError:
        raise InputClosed()


def parse_row(line, n):
    """
    한 줄을 n개의 숫자로 파싱한다.
    성공: (values, None) / 실패: (None, 오류 메시지)
    """
    tokens = line.split()
    if len(tokens) != n:
        return None, ("입력 형식 오류: 각 줄에 {0}개의 숫자를 공백으로 구분해 입력하세요. "
                      "(입력된 개수: {1})".format(n, len(tokens)))
    values = []
    for t in tokens:
        try:
            v = float(t)
        except ValueError:
            return None, "입력 형식 오류: '{0}'은(는) 숫자가 아닙니다. 숫자만 입력하세요.".format(t)
        if not math.isfinite(v):
            return None, "입력 형식 오류: '{0}'은(는) 유한한 숫자가 아닙니다.".format(t)
        values.append(v)
    return values, None


def read_matrix(name, n=3):
    """n줄을 입력받아 n×n 행렬로 저장한다. 잘못된 줄은 그 줄만 다시 입력받는다."""
    print("{0} ({1}줄 입력, 공백 구분)".format(name, n))
    matrix = make_matrix(n)
    r = 0
    while r < n:
        values, error = parse_row(read_line(), n)
        if error:
            print("  ! " + error)
            print("  ! {0}행을 다시 입력하세요.".format(r + 1))
            continue
        for c, v in enumerate(values):
            set_value(matrix, r, c, v)
        r += 1
    return matrix


def run_mode1():
    n = 3
    section("[1] 필터 입력")
    filter_a = read_matrix("필터 A", n)
    print()
    filter_b = read_matrix("필터 B", n)

    print()
    print("✓ 필터 저장 완료")
    print("필터 A:")
    print_matrix(filter_a)
    print("필터 B:")
    print_matrix(filter_b)

    section("[2] 패턴 입력")
    pattern = read_matrix("패턴", n)

    # 연산 (I/O와 분리)
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = (measure(mac, pattern, filter_a) + measure(mac, pattern, filter_b)) / 2
    winner = compare_scores(score_a, score_b)

    section("[3] MAC 결과")
    print("A 점수: " + fmt_score(score_a))
    print("B 점수: " + fmt_score(score_b))
    print("연산 시간(MAC 1회 평균/{0}회): {1:.4f} ms".format(REPEAT, avg_ms))
    if winner is None:
        print("판정: 판정 불가 (|A-B| = {0:.3e} < {1:g})".format(abs(score_a - score_b), EPSILON))
    else:
        print("판정: " + winner)

    section("[4] 성능 분석 (평균/{0}회)".format(REPEAT))
    print_perf_table([(n, avg_ms)])


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_json(path):
    """JSON을 읽는다. 실패 시 (None, 사유)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "파일을 찾을 수 없습니다: {0}".format(path)
    except json.JSONDecodeError as e:
        return None, "JSON 파싱 오류: {0}".format(e)
    except OSError as e:
        return None, "파일 읽기 오류: {0}".format(e)


def load_filters(raw_filters):
    """
    filters 섹션을 {N: {'Cross': matrix, 'X': matrix}} 형태로 정규화한다.
    문제가 있는 항목은 건너뛰고 메시지를 출력한다(프로그램은 계속 진행).
    """
    filters = {}
    if not isinstance(raw_filters, dict):
        print("✗ filters 섹션이 객체가 아닙니다")
        return filters

    for key in sorted(raw_filters, key=_size_sort_key):
        m = FILTER_KEY_RE.match(key)
        if not m:
            print("✗ {0}: 키 형식 오류 (size_N 형식이어야 함)".format(key))
            continue
        n = int(m.group(1))
        entry = raw_filters[key]
        if not isinstance(entry, dict):
            print("✗ {0}: 필터 묶음이 객체가 아닙니다".format(key))
            continue

        loaded = {}
        problems = []
        for raw_label, raw_matrix in entry.items():
            label = normalize_label(raw_label)
            if label is None:
                problems.append("알 수 없는 필터 라벨 '{0}'".format(raw_label))
                continue
            try:
                loaded[label] = to_matrix(raw_matrix, n)
            except ValueError as e:
                problems.append("{0} 필터 {1}".format(label, e))
        for label in STANDARD_LABELS:
            if label not in loaded and not any(p.startswith(label + " ") for p in problems):
                problems.append("{0} 필터 없음".format(label))

        filters[n] = loaded
        if problems:
            print("✗ {0:<8}필터 로드 문제: {1}".format(key, "; ".join(problems)))
        else:
            print("✓ {0:<8}필터 로드 완료 ({1})".format(key, ", ".join(STANDARD_LABELS)))
    return filters


def _size_sort_key(key):
    """size_N(_idx) 키는 숫자 순서로, 형식이 틀린 키는 맨 뒤에 둔다."""
    if FILTER_KEY_RE.match(key) or PATTERN_KEY_RE.match(key):
        return (0, [int(x) for x in re.findall(r"\d+", key)], key)
    return (1, [], key)


def analyze_case(key, case, filters):
    """
    패턴 1건을 분석한다. 어떤 문제가 있어도 예외를 밖으로 던지지 않고
    결과 dict(status='PASS'/'FAIL', reason=...)를 반환한다.
    """
    result = {"key": key, "status": "FAIL", "reason": None,
              "cross": None, "x": None, "decision": None, "expected": None}

    # 1) 키에서 N 추출
    m = PATTERN_KEY_RE.match(key)
    if not m:
        result["reason"] = "패턴 키 형식 오류 (size_{N}_{idx} 형식이어야 함)"
        return result
    n = int(m.group(1))

    if not isinstance(case, dict):
        result["reason"] = "패턴 항목이 객체가 아닙니다"
        return result

    # 2) expected 정규화
    expected = normalize_label(case.get("expected"))
    result["expected"] = expected
    if expected is None:
        result["reason"] = "알 수 없는 expected 라벨: {0!r}".format(case.get("expected"))
        return result

    # 3) 필터 선택: size_N
    filter_set = filters.get(n)
    if filter_set is None:
        result["reason"] = "size_{0} 필터가 없습니다".format(n)
        return result
    missing = [lb for lb in STANDARD_LABELS if lb not in filter_set]
    if missing:
        result["reason"] = "size_{0} 필터 불완전 (누락: {1})".format(n, ", ".join(missing))
        return result

    # 4) 패턴 크기 검증 (필터 크기 N과 일치해야 함)
    if "input" not in case:
        result["reason"] = "input 항목이 없습니다"
        return result
    try:
        pattern = to_matrix(case["input"], n)
    except ValueError as e:
        result["reason"] = "패턴-필터 크기/형식 불일치 (필터 {0}×{0}): {1}".format(n, e)
        return result

    # 5) MAC + 판정
    score_cross = mac(pattern, filter_set[LABEL_CROSS])
    score_x = mac(pattern, filter_set[LABEL_X])
    result["cross"], result["x"] = score_cross, score_x

    winner = compare_scores(score_cross, score_x)
    decision = {"A": LABEL_CROSS, "B": LABEL_X, None: "UNDECIDED"}[winner]
    result["decision"] = decision

    # 6) PASS / FAIL
    if decision == expected:
        result["status"] = "PASS"
    elif decision == "UNDECIDED":
        result["reason"] = ("동점(UNDECIDED) 처리 규칙에 따라 FAIL "
                            "(|Cross-X| = {0:.3e} < {1:g})".format(abs(score_cross - score_x), EPSILON))
    else:
        result["reason"] = "판정 {0} ≠ expected {1}".format(decision, expected)
    return result


def print_case(res):
    print("--- {0} ---".format(res["key"]))
    if res["cross"] is None:
        print("오류: " + res["reason"])
        print("결과: FAIL")
        return
    print("Cross 점수: " + fmt_score(res["cross"]))
    print("X 점수: " + fmt_score(res["x"]))
    tail = res["status"]
    if res["decision"] == "UNDECIDED":
        tail += " (동점 규칙)"
    print("판정: {0} | expected: {1} | {2}".format(res["decision"], res["expected"], tail))


def run_mode2(data_path):
    section("[1] 필터 로드")
    print("데이터 파일: {0}".format(data_path))
    data, error = load_json(data_path)
    if error:
        print("✗ " + error)
        print("모드 2를 종료합니다.")
        return
    if not isinstance(data, dict):
        print("✗ 최상위 구조가 객체가 아닙니다. 모드 2를 종료합니다.")
        return
    filters = load_filters(data.get("filters"))

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    raw_patterns = data.get("patterns")
    results = []
    if not isinstance(raw_patterns, dict):
        print("✗ patterns 섹션이 없거나 객체가 아닙니다")
    else:
        for key in sorted(raw_patterns, key=_size_sort_key):
            try:
                res = analyze_case(key, raw_patterns[key], filters)
            except Exception as e:  # 예상 못한 오류도 케이스 단위 FAIL로 격리
                res = {"key": key, "status": "FAIL", "cross": None,
                       "reason": "예상치 못한 오류: {0}".format(e)}
            results.append(res)
            print_case(res)

    section("[3] 성능 분석 (평균/{0}회)".format(REPEAT))
    print_perf_table(run_perf(PERF_SIZES, filters))

    section("[4] 결과 요약")
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] != "PASS"]
    print("총 테스트: {0}개".format(len(results)))
    print("통과: {0}개".format(len(passed)))
    print("실패: {0}개".format(len(failed)))
    if failed:
        print()
        print("실패 케이스:")
        for r in failed:
            print("- {0}: {1}".format(r["key"], r["reason"]))
    print()
    print('(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션 참고)')


# ---------------------------------------------------------------------------
# 모드 3: 보너스 (패턴 생성기 + 1차원 최적화 비교)
# ---------------------------------------------------------------------------
def run_mode3():
    section("[1] 패턴 생성기")
    while True:
        line = read_line("생성할 크기 N (3 이상 홀수, 엔터=5): ").strip()
        if line == "":
            n = 5
            break
        try:
            n = int(line)
        except ValueError:
            print("  ! 정수를 입력하세요.")
            continue
        if n < 3 or n % 2 == 0:
            print("  ! 3 이상의 홀수를 입력하세요.")
            continue
        break

    cross_n, x_n = generate_cross(n), generate_x(n)
    if n <= 13:
        print("Cross {0}×{0}:".format(n))
        print_matrix(cross_n)
        print("X {0}×{0}:".format(n))
        print_matrix(x_n)
    for name, pattern in ((LABEL_CROSS, cross_n), (LABEL_X, x_n)):
        sa, sb = mac(pattern, cross_n), mac(pattern, x_n)
        w = {"A": LABEL_CROSS, "B": LABEL_X, None: "UNDECIDED"}[compare_scores(sa, sb)]
        print("생성 패턴 {0:<5} → Cross 점수 {1:g}, X 점수 {2:g}, 판정 {3}".format(name, sa, sb, w))

    section("[2] 최적화 전/후 성능 비교 (평균/{0}회, 동일 입력)".format(REPEAT))
    table = []
    for size in BONUS_SIZES:
        p2, f2 = generate_cross(size), generate_cross(size)
        p1, f1 = flatten(p2), flatten(f2)          # 변환은 측정 구간 밖
        assert abs(mac(p2, f2) - mac_flat(p1, f1)) < EPSILON
        t2 = measure(mac, p2, f2)
        t1 = measure(mac_flat, p1, f1)
        gain = (t2 - t1) / t2 * 100 if t2 > 0 else 0.0
        table.append(["{0}×{0}".format(size), str(size * size), "{0:.4f}".format(t2),
                      "{0:.4f}".format(t1), "{0:.1f}%".format(gain)])
    print_table(["크기(N×N)", "연산 횟수(N²)", "2차원(ms)", "1차원(ms)", "개선율"], table, [10, 16, 14, 14, 10])


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def choose_mode():
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 보너스: 패턴 생성기 + 최적화 비교")
    while True:
        choice = read_line("선택: ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("  ! 1, 2, 3 중 하나를 입력하세요.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mini NPU Simulator")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                        help="모드 2에서 사용할 JSON 경로 (기본: main.py 옆 data.json)")
    args = parser.parse_args(argv)

    print("=== Mini NPU Simulator ===")
    print()
    try:
        choice = choose_mode()
        if choice == "1":
            run_mode1()
        elif choice == "2":
            run_mode2(args.data)
        else:
            run_mode3()
    except InputClosed:
        print("\n입력이 종료되어 프로그램을 마칩니다.")
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")


if __name__ == "__main__":
    main()
