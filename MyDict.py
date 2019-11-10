class my_dictionary(dict): 

    # __init__ function 
    def __init__(self): 
        self = dict() 
		
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 

    def GetValue(self,key):
        return self[key]

    def Print(self):
        print(self)
