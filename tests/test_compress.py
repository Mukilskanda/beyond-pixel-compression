from textpath.compress import compress_markdown


def test_dedup_drops_repeated_line_keeps_first_occurrence():
    markdown = "Acme Corp Annual Report\n\nSome unique content here.\n\nAcme Corp Annual Report"
    out = compress_markdown(markdown)
    assert out.count("Acme Corp Annual Report") == 1
    assert "Some unique content here." in out


def test_dedup_is_case_insensitive():
    markdown = "Page Header\n\nbody text\n\nPAGE HEADER"
    out = compress_markdown(markdown)
    assert out.lower().count("page header") == 1


def test_collapses_blank_paragraphs():
    markdown = "First paragraph.\n\n\n\n   \n\nSecond paragraph."
    out = compress_markdown(markdown)
    assert "\n\n\n" not in out
    assert "First paragraph." in out
    assert "Second paragraph." in out


def test_dedup_false_keeps_duplicate_lines():
    markdown = "Repeat me\n\nRepeat me"
    out = compress_markdown(markdown, dedup_lines=False)
    assert out.count("Repeat me") == 2


def test_no_content_lost_within_a_paragraph_beyond_exact_duplicates():
    markdown = "Line one\nLine two\nLine one"
    out = compress_markdown(markdown)
    assert "Line one" in out
    assert "Line two" in out
    assert out.count("Line one") == 1


def test_empty_input_returns_empty_string():
    assert compress_markdown("") == ""
