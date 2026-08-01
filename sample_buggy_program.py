def run(candidate):
    total = 0
    for value in candidate:
        if(value < 0):
           total += value
    return total
