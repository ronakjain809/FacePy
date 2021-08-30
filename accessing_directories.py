import os

print(os.listdir(r"D:\\"))

dictionary = {}

liSt = os.listdir(r"D:\\")

i = 0

for name in liSt:

    dictionary[name] = i
    i = i + 1

print(dictionary)

