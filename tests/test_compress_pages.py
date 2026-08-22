from textpath.compress import compress_markdown, compress_pages


def test_cross_page_dedup_removes_repeated_header_from_later_page():
    pages = [
        "ACME CORP REPORT\n\nSection 1 body text.",
        "ACME CORP REPORT\n\nSection 2 body text.",
    ]

    result = compress_pages(pages)

    assert "ACME CORP REPORT" in result[0]  # first occurrence kept
    assert "ACME CORP REPORT" not in result[1]  # later duplicate dropped
    assert "Section 1 body text." in result[0]
    assert "Section 2 body text." in result[1]  # distinct content untouched


def test_cross_page_dedup_keeps_first_page_fully_intact():
    pages = ["Unique header\n\nUnique body"]
    result = compress_pages(pages)
    assert result[0] == compress_markdown(pages[0])


def test_cross_page_dedup_is_order_dependent():
    """Page order determines which occurrence survives — first page's line
    is always the one kept, matching this project's reading-order
    convention elsewhere (textpath/compress.py's within-page dedup)."""
    pages = ["Repeated line\n\nFirst page unique", "Repeated line\n\nSecond page unique"]
    result = compress_pages(pages)
    assert "Repeated line" in result[0]
    assert "Repeated line" not in result[1]


def test_cross_page_dedup_does_not_touch_genuinely_distinct_content():
    pages = ["Total: $100", "Total: $200"]
    result = compress_pages(pages)
    # Different lines, not duplicates of each other — both survive.
    assert "Total: $100" in result[0]
    assert "Total: $200" in result[1]


def test_cross_page_dedup_can_over_delete_a_coincidentally_identical_value():
    """Documented, real limitation (exact-string, not semantic dedup): a
    genuinely meaningful value that happens to repeat verbatim across pages
    (e.g. the same dollar total) is dropped just like real boilerplate
    would be. This test pins the current (accepted) behavior rather than
    silently discovering it later."""
    pages = ["Total: $100", "Total: $100"]  # same value, could be meaningful on both pages
    result = compress_pages(pages)
    assert "Total: $100" in result[0]
    assert "Total: $100" not in result[1]  # over-deleted, by design tradeoff


def test_cross_page_dedup_empty_pages_list_returns_empty():
    assert compress_pages([]) == []


def test_dedup_lines_false_disables_cross_page_dedup_too():
    pages = ["Repeated\n\nA", "Repeated\n\nB"]
    result = compress_pages(pages, dedup_lines=False)
    assert "Repeated" in result[0]
    assert "Repeated" in result[1]
