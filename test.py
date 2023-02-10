class car:
    def __init__(self,brand_name,colour,model):
        self.brand_name=brand_name
        self.colour=colour
        self.model=model

    def myfunc(self):
        print(f"My car name is {self.brand_name} of {self.colour} colour of {self.model} model")
    def __str__(self):
        return "brand_name:"+self.brand_name+",colour:"+self.colour+'model:'+str(self.model)
p1=car("Baleno",'Dark blue',2019)
p2=car("brezza",'black',2019)
p1.myfunc()
print(p1)
print(p1+p2)