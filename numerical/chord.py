def fx(x):
    return x**3 - x - 2




def main():
    a = 1.0
    b = 2.0
    E = 0.001
    x0 = a
    x1 = x0-(fx(x0) * (b - x0)) / (fx(b) - fx(x0))
    while True:
        x1 = x0 - (fx(x0) * (b - x0)) / (fx(b) - fx(x0))
        if x1-x0 < E:
            break
        else:
            x0 = x1

    print( x1, abs(x1-x0) ) 
        



