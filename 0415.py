class MyCalss:
    def __init__(self,num):
        self.numbers1 = [n for n in range(num)]
        self.numbers2 = [n**2 for n in range(num)]

    def __getitem__(self, idx): # 인덱스기능
        return self.numbers1[idx]

    def __len__(self): # 길이 보는 기능
        return len(self.numbers1)


obj = MyCalss(10)
print(len(obj))
