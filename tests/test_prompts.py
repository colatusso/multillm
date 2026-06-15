from multillm.prompts import compose_system


def test_base_and_role():
    assert compose_system("voz", "papel") == "voz\n\npapel"


def test_strips_and_skips_empty():
    assert compose_system("  voz  ", "") == "voz"
    assert compose_system("", "papel") == "papel"
    assert compose_system("", "") == ""
    assert compose_system("   ", "  ") == ""


def test_custom_sep():
    assert compose_system("a", "b", sep=" | ") == "a | b"
