#pip install chatterbot

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
Chinese_bot = ChatBot("Training demo") #创建一个新的实例
Chinese_bot.set_trainer(ListTrainer)
Chinese_bot.train([
    '亲，在吗？',
    '亲，在呢',
    '这件衣服的号码大小标准吗？',
    '亲，标准呢，请放心下单吧。',
    '有红色的吗？',
    '有呢，目前有白红蓝3种色调。',
])

# 测试一下
question = '亲，在吗'
print(question)
response = Chinese_bot.get_response(question)
print(response)
print("\n")
question = '有红色的吗？'
print(question)
response = Chinese_bot.get_response(question)
print(response)