dic = {"apple":3,"orange":2,"banana":2}
print(dic.items())
lis = list(dic.items())
lis.sort(key=lambda item:(item[1],item[0]))
print(lis)
