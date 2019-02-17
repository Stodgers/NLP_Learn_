import genius
text = '卧槽泥马勒戈壁的煞笔东西你像根草'
'''
text 第一个参数为需要分词的字。
use_break 代表对分词结构进行打断处理，默认值 True。
use_combine 代表是否使用字典进行词合并，默认值 False。
use_tagging 代表是否进行词性标注，默认值 True。
use_pinyin_segment 代表是否对拼音进行分词处理，默认值 True。
'''
seg_list = genius.seg_text(
    text,
    use_combine=True,
    use_pinyin_segment=True,
    use_tagging=True,
    use_break=True
)

print(["/".join(i) for i in seg_list])