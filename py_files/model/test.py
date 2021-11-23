

class Test():
    def __init__(self, input_dim = 13, num_classes = 7):
        super().__init__()

        self.para = input_dim
        def do(para):
            return para
        self.test = do(self.para)

    
    def print_it(self):
        print(self.test)


bla =  Test(input_dim =15)
bla.print_it()