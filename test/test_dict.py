dict = {"a": 1, "b": 2, "c": 3}

print(dict)

print(dict.keys())

print(dict.values())

print(dict.items())

print(dict.get("a"))

print(dict.get("d"))

print(dict.get("d", 4))

print(dict.pop("a"))

dict2 = {"love": 1, **dict}

print(dict2)
