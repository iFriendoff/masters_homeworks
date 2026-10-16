def midd(a, b):
    return (a + b) / 2

def f1(x):
    return x**3 - x - 2

def main():
    a = 1.0
    b = 2.0
    E = 0.001

    while abs(f1(midd(a, b))) > E:
        if f1(a) * f1(midd(a, b)) < 0:
            b = midd(a, b)
        else:
            a = midd(a, b)

    print( midd(a, b), f1(midd(a, b)) )

if __name__ == "__main__":
    main()