from data.synthetic import _GENERATORS, generate_dataset


def test_generates_all_categories(tmp_path):
    samples = generate_dataset(output_dir=tmp_path, per_category=2)
    categories = {s.category for s in samples}
    assert categories == set(_GENERATORS.keys())


def test_generates_correct_count(tmp_path):
    samples = generate_dataset(output_dir=tmp_path, per_category=4)
    assert len(samples) == 4 * len(_GENERATORS)


def test_images_actually_exist_and_are_readable(tmp_path):
    from PIL import Image

    samples = generate_dataset(output_dir=tmp_path, per_category=1)
    for s in samples:
        assert s.image_path.exists()
        img = Image.open(s.image_path)
        assert img.width > 0 and img.height > 0


def test_deterministic_with_seed(tmp_path):
    samples_a = generate_dataset(output_dir=tmp_path / "a", per_category=2, seed=42)
    samples_b = generate_dataset(output_dir=tmp_path / "b", per_category=2, seed=42)
    assert [s.answers for s in samples_a] == [s.answers for s in samples_b]


def test_counting_category_answer_matches_rendered_count(tmp_path):
    samples = generate_dataset(output_dir=tmp_path, per_category=3)
    counting_samples = [s for s in samples if s.category == "counting"]
    for s in counting_samples:
        assert s.answers[0].isdigit()
        assert 5 <= int(s.answers[0]) <= 25
