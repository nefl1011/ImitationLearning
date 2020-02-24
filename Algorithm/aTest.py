from scipy.stats import stats

if __name__ == '__main__':
    a = [0, 10, 2, 3, 3, 2, 3]
    m = stats.mode(a)
    print(m)
    print(m[0][0])