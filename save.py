def saveCSV(X, Y):
    F = open('optimumDataset.csv', 'w')
    for x, y in zip(X, Y):
        for each in x:
            F.write(str(each) + ',')
        F.write(str(int(y)) + '\n')
    F.close()


def saveBestK(K):
    F = open('selectedIndex.csv', 'w')
    ensure = True
    for i in K:
        if ensure:
            F.write(str(i))
        else:
            F.write(','+str(i))
        ensure = False
    F.close()

def saveFeatures(tracking):
    F = open('trackingFeaturesStructure.txt', 'w')
    ensure = True
    for i in tracking:
        if ensure:
            F.write(str(i))
        else:
            F.write(',' + str(i))
        ensure = False
    F.close()




