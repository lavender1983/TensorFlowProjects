# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep6.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-11 13:56
# Remarks  : Unicode 字符串
import tensorflow as tf

text_utf8 = tf.constant(u"语言处理")
print(text_utf8)
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
print(text_utf16be)
text_chars = tf.constant([ord(char) for char in u"语言处理"])
print(text_chars)

print(tf.strings.unicode_decode(text_utf8, input_encoding="UTF-8"))
print(tf.strings.unicode_encode(text_chars, output_encoding="UTF-8"))
print(tf.strings.unicode_transcode(text_utf8, input_encoding="UTF-8", output_encoding="UTF-16-BE"))

batch_utf8 = [s.encode('UTF-8') for s in [u'hÃllo', u'What is the weather tomorrow', u'Göödnight', u'😊']]
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8, input_encoding="UTF-8")

for sentence_chars in batch_chars_ragged.to_list():
    print(sentence_chars)

batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
print(batch_chars_padded)
batch_chars_sparse = batch_chars_ragged.to_sparse()
print(batch_chars_sparse)

text_animals = tf.strings.unicode_encode([[99, 97, 116], [100, 111, 103], [99, 111, 119]], output_encoding="UTF-8")
print(text_animals)

print(tf.strings.unicode_encode(batch_chars_ragged, output_encoding="UTF-8"))

print(tf.strings.unicode_encode(tf.RaggedTensor.from_sparse(batch_chars_sparse), output_encoding="UTF-8"))

print(tf.strings.unicode_encode(tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1), output_encoding="UTF-8"))

# 运算
thanks = u'Thanks 😊'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit="UTF8_CHAR").numpy()
print("{} bytes; {} UTF-8 characters".format(num_bytes, num_chars))

print(tf.strings.substr(thanks, pos=7, len=1).numpy())
print(tf.strings.substr(thanks, pos=8, len=1, unit="UTF8_CHAR").numpy())
print(tf.strings.split(thanks, "UTF-8").numpy())

codepoints, offsets = tf.strings.unicode_decode_with_offsets(u"🎈🎉🎊", 'UTF-8')

for codepoint, offset in zip(codepoints.numpy(), offsets.numpy()):
    print("At byte offset {}: codepoint {}".format(offset, codepoint))

uscript = tf.strings.unicode_script([33464, 1041])
print(uscript.numpy())

# 简单分词
sentence_texts = [u'Hello, world.', u'世界こんにちは']
# 将句子解码为字符码位，然后查找每个字符的字符体系标识符
sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts, "UTF-8")
print(sentence_char_codepoint)

sentence_char_script = tf.strings.unicode_script(sentence_char_codepoint)
print(sentence_char_script)

# 添加词边界
sentence_char_starts_word = tf.concat([
    tf.fill([sentence_char_script.nrows(), 1], True),
    tf.not_equal(sentence_char_script[:, 1:], sentence_char_script[:, :-1])
],
                                      axis=1)

print(sentence_char_starts_word)

word_starts = tf.squeeze(tf.where(sentence_char_starts_word.values), axis=1)
print(word_starts)

sentence_word_char_codepoint = tf.RaggedTensor.from_row_starts(values=sentence_char_codepoint.values,
                                                               row_starts=word_starts)
print(sentence_word_char_codepoint)

ls = tf.strings.unicode_encode(sentence_word_char_codepoint, 'UTF-8')
print(ls)
