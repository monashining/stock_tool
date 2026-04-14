from line_quick_reply import build_quick_reply_items, display_code_for_quick_reply


def test_display_code():
    assert display_code_for_quick_reply("3037") == "3037"
    assert display_code_for_quick_reply("3037.TW") == "3037"
    assert display_code_for_quick_reply("") == "2330"


def test_build_quick_reply_items_shape():
    items = build_quick_reply_items("3189")
    assert len(items) == 6
    assert items[0]["action"]["text"] == "查 3189"
    assert items[1]["action"]["text"] == "速查 3189"
    assert items[2]["action"]["text"] == "完整 3189"
    assert items[5]["action"]["text"] == "help"
