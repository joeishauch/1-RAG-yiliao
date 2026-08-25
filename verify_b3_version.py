from pathlib import Path
from tempfile import TemporaryDirectory
import json
from utils.kg_builder import build_or_load, graph_cache_matches

rec1 = {"questions": ["流感的临床表现"], "answers": ["发热；咳嗽"]}
rec2 = {"questions": ["流感的就诊科室"], "answers": ["呼吸内科"]}

with TemporaryDirectory() as d:
    source = Path(d) / "source.jsonl"
    output = Path(d) / "graph.pkl"
    source.write_text(
        json.dumps(rec1, ensure_ascii=False) + "\n"
        + json.dumps(rec2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    first = build_or_load(str(source), str(output), sample=None)
    second = build_or_load(str(source), str(output), sample=None)
    assert first.graph.get("source_sha256")
    assert second.graph.get("source_sha256") == first.graph.get("source_sha256")
    assert graph_cache_matches(second, second.graph["source_sha256"], None, source)
    before = second.graph["source_sha256"]
    source.write_text(
        source.read_text(encoding="utf-8")
        + json.dumps({"questions": ["流感的并发症"], "answers": ["肺炎"]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    third = build_or_load(str(source), str(output), sample=None)
    assert third.graph["source_sha256"] != before
    print("KG_VERSION_TEST_OK")
