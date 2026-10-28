from callcenter.rag import FaqIndex


def test_e5_prefixes_are_applied(embedder, faq):
    index = FaqIndex(embedder)
    index.add(faq)
    index.search("駐車場はある？")
    assert all(t.startswith("passage: ") for t in embedder.calls[: len(faq)])
    assert embedder.calls[-1].startswith("query: ")


def test_search_returns_best_match_first(embedder, faq):
    index = FaqIndex(embedder)
    index.add(faq)
    hits = index.search("駐車場はありますか", k=3)
    assert hits[0].entry.id == "access-01"
    assert hits[0].score > 0.99
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_empty_index_returns_nothing(embedder):
    assert FaqIndex(embedder).search("何でも") == []
