def jquicksort(lst):
    if len(lst) < 2:
        return lst
    smaller = [x for x in lst[1:] if x < lst[0]]
    larger = [x for x in lst[1:] if x >= lst[0]]
    return jquicksort(smaller) + [lst[0]] + jquicksort(larger)

if __name__ == '__main__':
    print jquicksort([1, 3, 2])
    print jquicksort([18, 17, 74, 92, 1030, 3, 42])
