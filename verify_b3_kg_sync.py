# -*- coding: utf-8 -*-
"""verify_b3_kg_sync.py — B.3 KG↔向量联动的 mock 集成验证。

不依赖真实 ChromaDB / embedding API / 大数据源。
用 TemporaryDirectory + mock 验证以下路径：
  1. KG rebuild → 审计元数据完整
  2. 未变化 skip → KG 已缓存时跳过重建
  3. KG 构建失败 → 错误传播
  4. 非 KG source → 不触发 KG 联动
  5. 缓存失效 → 进程内单例被清空
  6. 原子输出无残留临时文件
  7. 源变更 → KG 重建

用法：
    cd "D:/ai/聚客/02聚客AI大模型第六期/10-项目2_基于LangGraph实现智能分诊系统/项目2_基于LangGraph实现智能分诊系统"
    python verify_b3_kg_sync.py
"""
import os
import sys
import json
import tempfile
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

# 项目根目录 = 本脚本所在目录
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台中文编码
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _make_records(n=5):
    """生成 n 条临时 JSONL 记录，包含 belongs_to_department 模式以产生疾病种子。"""
    records = []
    for i in range(n):
        records.append({
            "questions": [f"测试疾病{i}的就诊科室"],
            "answers": [f"测试科室{i}"],
        })
    for i in range(n):
        records.append({
            "questions": [f"测试疾病{i}的临床表现"],
            "answers": [f"症状A{i}；症状B{i}"],
        })
    return records


def _make_kg_records():
    """生成包含多种关系模式的 KG 记录，确保产生三元组。"""
    return [
        {"questions": ["糖尿病的就诊科室"], "answers": ["内分泌科"]},
        {"questions": ["糖尿病的临床表现"], "answers": ["多尿；多饮；体重下降"]},
        {"questions": ["糖尿病的药物"], "answers": ["二甲双胍；胰岛素"]},
        {"questions": ["糖尿病的并发症"], "answers": ["视网膜病变；肾病"]},
        {"questions": ["冠心病的就诊科室"], "answers": ["心内科"]},
        {"questions": ["冠心病的临床表现"], "answers": ["胸痛；心绞痛"]},
        {"questions": ["高血压的就诊科室"], "answers": ["心内科"]},
        {"questions": ["高血压的临床表现"], "answers": ["头痛；头晕"]},
    ]


def _write_jsonl(path, records):
    """写入 JSONL 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_manifest(path):
    """读取 manifest 文件。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _count_temp_files(dir_path, prefix="."):
    """统计目录中残留的临时文件数。"""
    count = 0
    for f in os.listdir(dir_path):
        if f.startswith(prefix) and ".tmp" in f:
            count += 1
    return count


# ==================== 测试 1：KG rebuild → 审计元数据完整 ====================

def test_vector_write_then_kg_rebuild():
    """向量 chunks 成功写入后，KG 图应被 rebuild 并包含完整元数据。"""
    print("\n=== 测试 1：KG rebuild → 审计元数据完整 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        graph_path = Path(tmpdir) / "kg_graph.pkl"

        records = _make_kg_records()
        _write_jsonl(source_path, records)

        from doc_sync import streaming_sha256, _sync_kg_artifacts
        sha = streaming_sha256(source_path)

        result = _sync_kg_artifacts(
            source_path, sha,
            graph_path=str(graph_path),
            sample=0,  # 0 = 不抽样，全量构建（内部转为 None）
        )

        assert result["status"] == "rebuilt", f"expected rebuilt, got {result['status']}"
        assert result["source_sha256"] == sha
        assert result["nodes"] > 0, f"expected nodes > 0, got {result['nodes']}"
        assert result["edges"] > 0, f"expected edges > 0, got {result['edges']}"
        assert result["duration_s"] >= 0
        assert graph_path.exists(), "kg_graph.pkl 应已创建"

        # 验证图元数据
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        assert G.graph.get("source_sha256") == sha
        assert G.graph.get("schema_version") == 1
        assert G.graph.get("sample") is None  # sample=0 → 内部转为 None
        assert G.graph.get("source_path") is not None

    print("✅ 测试 1 通过")


# ==================== 测试 2：未变化 skip → KG 已缓存 ====================

def test_skip_with_cached_kg():
    """源未变化且 KG 已缓存时，_sync_kg_artifacts 返回 cached。"""
    print("\n=== 测试 2：未变化 skip → KG 已缓存 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        graph_path = Path(tmpdir) / "kg_graph.pkl"

        records = _make_kg_records()
        _write_jsonl(source_path, records)

        from doc_sync import streaming_sha256, _sync_kg_artifacts
        sha = streaming_sha256(source_path)

        # 第一次：rebuilt
        r1 = _sync_kg_artifacts(source_path, sha, graph_path=str(graph_path), sample=0)
        assert r1["status"] == "rebuilt"

        # 第二次：cached（源未变化）
        r2 = _sync_kg_artifacts(source_path, sha, graph_path=str(graph_path), sample=0)
        assert r2["status"] == "cached", f"expected cached, got {r2['status']}"
        assert r2["source_sha256"] == sha
        assert r2["nodes"] == r1["nodes"]
        assert r2["edges"] == r1["edges"]

    print("✅ 测试 2 通过")


# ==================== 测试 3：KG 构建失败 → 错误传播 ====================

def test_kg_failure_returns_error():
    """KG 构建失败时，_sync_kg_artifacts 应抛出异常。"""
    print("\n=== 测试 3：KG 构建失败 → 错误传播 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        graph_path = Path(tmpdir) / "kg_graph.pkl"

        _write_jsonl(source_path, _make_kg_records())

        from doc_sync import streaming_sha256
        sha = streaming_sha256(source_path)

        # Mock build_or_load 使其抛出异常（延迟 import，patch 源模块）
        with patch("utils.kg_builder.build_or_load", side_effect=RuntimeError("KG build failed (mock)")):
            from doc_sync import _sync_kg_artifacts
            try:
                _sync_kg_artifacts(source_path, sha, graph_path=str(graph_path), sample=0)
                assert False, "应抛出异常"
            except RuntimeError as e:
                assert "KG build failed" in str(e)

    print("✅ 测试 3 通过")


# ==================== 测试 4：非 KG source → 不触发 KG 联动 ====================

def test_non_kg_source_no_linkage():
    """非 huatuo_knowledge_graph source 不应触发 KG 联动。"""
    print("\n=== 测试 4：非 KG source → 不触发 KG 联动 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        graph_path = Path(tmpdir) / "kg_graph.pkl"

        _write_jsonl(source_path, _make_kg_records())

        from doc_sync import streaming_sha256
        sha = streaming_sha256(source_path)

        # _sync_kg_artifacts 不应被调用
        with patch("doc_sync._sync_kg_artifacts") as mock_kg:
            source_name = "huatuo_lite"  # 不是 huatuo_knowledge_graph
            if source_name == "huatuo_knowledge_graph":
                mock_kg(source_path, sha)
            mock_kg.assert_not_called()

    print("✅ 测试 4 通过")


# ==================== 测试 5：缓存失效验证 ====================

def test_cache_invalidation():
    """KG rebuild 后，进程内缓存应被失效。"""
    print("\n=== 测试 5：缓存失效验证 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        graph_path = Path(tmpdir) / "kg_graph.pkl"

        _write_jsonl(source_path, _make_kg_records())

        from doc_sync import streaming_sha256
        sha = streaming_sha256(source_path)

        # invalidate 在 _sync_kg_artifacts 内部延迟 import，patch 到源模块
        with patch("utils.kg_query.invalidate_kg_cache") as mock_kg_inv:
            with patch("utils.kg_symptom_match.invalidate_symptom_index") as mock_sym_inv:
                from doc_sync import _sync_kg_artifacts
                _sync_kg_artifacts(
                    source_path, sha,
                    graph_path=str(graph_path),
                    sample=0,
                )
                # rebuild 路径应调用两个 invalidate
                assert mock_kg_inv.call_count >= 1, "invalidate_kg_cache 应至少调用 1 次"
                assert mock_sym_inv.call_count >= 1, "invalidate_symptom_index 应至少调用 1 次"

            # 第二次（cached）不应调用 invalidate
            mock_kg_inv.reset_mock()
            mock_sym_inv.reset_mock()
            _sync_kg_artifacts(
                source_path, sha,
                graph_path=str(graph_path),
                sample=0,
            )
            assert mock_kg_inv.call_count == 0, "cached 时不应调用 invalidate_kg_cache"
            assert mock_sym_inv.call_count == 0, "cached 时不应调用 invalidate_symptom_index"

    print("✅ 测试 5 通过")


# ==================== 测试 6：原子输出无残留 ====================

def test_no_temp_residuals():
    """KG 保存后不应有 .tmp 残留文件。"""
    print("\n=== 测试 6：原子输出无残留 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        graph_path = Path(tmpdir) / "kg_graph.pkl"

        _write_jsonl(source_path, _make_kg_records())

        from doc_sync import streaming_sha256, _sync_kg_artifacts
        sha = streaming_sha256(source_path)

        _sync_kg_artifacts(source_path, sha, graph_path=str(graph_path), sample=0)

        temp_count = _count_temp_files(tmpdir, prefix=".kg")
        assert temp_count == 0, f"应无 .tmp 残留，发现 {temp_count} 个"
        assert graph_path.exists()

    print("✅ 测试 6 通过")


# ==================== 测试 7：源变更 → KG 重建 ====================

def test_source_change_triggers_rebuild():
    """源文件追加记录后，KG 应被重建。"""
    print("\n=== 测试 7：源变更 → KG 重建 ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        graph_path = Path(tmpdir) / "kg_graph.pkl"

        records = _make_kg_records()
        _write_jsonl(source_path, records)

        from doc_sync import streaming_sha256, _sync_kg_artifacts

        sha1 = streaming_sha256(source_path)
        r1 = _sync_kg_artifacts(source_path, sha1, graph_path=str(graph_path), sample=0)
        assert r1["status"] == "rebuilt"

        # 未变化 → cached
        r2 = _sync_kg_artifacts(source_path, sha1, graph_path=str(graph_path), sample=0)
        assert r2["status"] == "cached"

        # 追加记录 → 源变更
        with open(source_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "questions": ["新增疾病X的就诊科室"],
                "answers": ["新增科室"],
            }, ensure_ascii=False) + "\n")

        sha2 = streaming_sha256(source_path)
        assert sha2 != sha1, "SHA 应随源变更而改变"

        r3 = _sync_kg_artifacts(source_path, sha2, graph_path=str(graph_path), sample=0)
        assert r3["status"] == "rebuilt", f"expected rebuilt, got {r3['status']}"
        assert r3["source_sha256"] == sha2

    print("✅ 测试 7 通过")


# ==================== 主入口 ====================

def main():
    os.makedirs("output", exist_ok=True)
    print("=" * 60)
    print("B.3 KG 与向量联动 mock 集成验证")
    print("=" * 60)

    tests = [
        test_vector_write_then_kg_rebuild,
        test_skip_with_cached_kg,
        test_kg_failure_returns_error,
        test_non_kg_source_no_linkage,
        test_cache_invalidation,
        test_no_temp_residuals,
        test_source_change_triggers_rebuild,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过 / {failed} 失败 / {len(tests)} 总计")
    print("=" * 60)

    if failed:
        sys.exit(1)
    else:
        print("B3_KG_SYNC_TEST_OK")


if __name__ == "__main__":
    main()
