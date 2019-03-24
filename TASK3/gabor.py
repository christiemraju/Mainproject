import cv2
import numpy as np
import math
import _utils
import os.path

def getFilterImages(filters, img):
    featureImages = []
    for filter in filters:
        kern, params = filter
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        featureImages.append(fimg)
    return featureImages

def filterSelection(featureImages, threshold, img, howManyFilterImages):

    idEnergyList = []
    id = 0
    height, width = img.shape
    for featureImage in featureImages:
        thisEnergy = 0.0
        for x in range(height):
            for y in range(width):
                thisEnergy += pow(np.abs(featureImage[x][y]), 2)
        idEnergyList.append((thisEnergy, id))
        id += 1
    E = 0.0
    for E_i in idEnergyList:
        E += E_i[0]
    sortedlist = sorted(idEnergyList, key=lambda energy: energy[0], reverse = True)

    tempSum = 0.0
    RSquared = 0.0
    added = 0
    outputFeatureImages = []
    while ((RSquared < threshold) and (added < howManyFilterImages)):
        tempSum += sortedlist[added][0]
        RSquared = (tempSum/E)
        outputFeatureImages.append(featureImages[sortedlist[added][1]])
        added += 1
    return outputFeatureImages


def build_filters(lambdas, ksize, gammaSigmaPsi):

    filters = []
    thetas = []
    thetas.extend([0, 45, 90, 135])
    thetasInRadians = [np.deg2rad(x) for x in thetas]

    for lamb in lambdas:
        for theta in thetasInRadians:
            params = {'ksize': (ksize, ksize), 'sigma': gammaSigmaPsi[1], 'theta': theta, 'lambd': lamb,
                   'gamma':gammaSigmaPsi[0], 'psi': gammaSigmaPsi[2], 'ktype': cv2.CV_64F}
            kern = cv2.getGaborKernel(**params)
            kern /= 1.5 * kern.sum()
            filters.append((kern, params))
    return filters


def getLambdaValues(img):
    height, width = img.shape


    print width
    max = (width/4) * math.sqrt(2)
    print max
    min = 4 * math.sqrt(2)
    temp = min
    radialFrequencies = []


    while(temp < max):
        radialFrequencies.append(temp)
        temp = temp * 2

    radialFrequencies.append(max)
    lambdaVals = []
    for freq in radialFrequencies:
        lambdaVals.append(width/freq)
    return lambdaVals


def nonLinearTransducer(img, gaborImages, L, sigmaWeight, filters):

    alpha_ = 0.25
    featureImages = []
    count = 0
    for gaborImage in gaborImages:


        avgPerRow = np.average(gaborImage, axis=0)
        avg = np.average(avgPerRow, axis=0)
        gaborImage = gaborImage.astype(float) - avg


        if int(cv2.__version__[0]) >= 3:
            gaborImage = cv2.normalize(gaborImage, gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            gaborImage = cv2.normalize(gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        height, width = gaborImage.shape
        copy = np.zeros(img.shape)
        for row in range(height):
            for col in range(width):
                #centralPixelTangentCalculation_bruteForce(gaborImage, copy, row, col, alpha, L)
                copy[row][col] = math.fabs(math.tanh(alpha_ * (gaborImage[row][col])))

        # now apply smoothing
        copy, destroyImage = applyGaussian(copy, L, sigmaWeight, filters[count])
        if(not destroyImage):
            featureImages.append(copy)
        count += 1

    return featureImages


def applyGaussian(gaborImage, L, sigmaWeight, filter):

    height, N_c = gaborImage.shape

    nparr = np.array(filter[0])
    u_0 = nparr.mean(axis=0)
    u_0 = u_0.mean(axis=0)

    destroyImage = False
    sig = 1
    if (u_0 < 0.000001):
        print("div by zero occured for calculation:")
        print("sigma = sigma_weight * (N_c/u_0), sigma will be set to zero")
        print("removing potential feature image!")
        destroyImage = True
    else:
        sig = sigmaWeight * (N_c / u_0)

    return cv2.GaussianBlur(gaborImage, (L, L), sig), destroyImage

def removeFeatureImagesWithSmallVariance(featureImages, threshold):
    toReturn =[]
    for image in featureImages:
        if(np.var(image) > threshold):
            toReturn.append(image)

    return toReturn


def runGabor(a,o,k,gk,M,spw,gamma,sigma,psi,vt,fi,R,siw,c,i):

    if(not os.path.isfile(a)):
        print(a, " is not a file!")
        exit(0)

    printlocation = os.path.dirname(os.path.abspath(o))
    _utils.deleteExistingSubResults(printlocation)

    M_transducerWindowSize = M
    if((M_transducerWindowSize % 2) == 0):
        print('Gaussian window size not odd, using next odd number')
        M_transducerWindowSize += 1

    k_clusters = k
    k_gaborSize = gk

    spatialWeight = spw
    gammaSigmaPsi = []
    gammaSigmaPsi.append(gamma)
    gammaSigmaPsi.append(sigma)
    gammaSigmaPsi.append(psi)
    variance_Threshold = vt
    howManyFeatureImages = fi
    R_threshold = R
    sigmaWeight = siw
    greyOutput = c
    printIntermediateResults = i

    if int(cv2.__version__[0]) >= 3:
        img = cv2.imread(a, 0)
    else:
        img = cv2.imread(a, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    lambdas = getLambdaValues(img)
    filters = build_filters(lambdas, k_gaborSize, gammaSigmaPsi)

    print("Gabor kernels created, getting filtered images")
    filteredImages = getFilterImages(filters, img)
    filteredImages = filterSelection(filteredImages, R_threshold, img, howManyFeatureImages)
    if(printIntermediateResults):
        _utils.printFeatureImages(filteredImages, "filter", printlocation)

    print("Applying nonlinear transduction with Gaussian smoothing")
    featureImages = nonLinearTransducer(img, filteredImages, M_transducerWindowSize, sigmaWeight, filters)
    featureImages = removeFeatureImagesWithSmallVariance(featureImages, variance_Threshold)

    if (printIntermediateResults):
        _utils.printFeatureImages(featureImages, "feature", printlocation)

    featureVectors = _utils.constructFeatureVectors(featureImages, img)
    featureVectors = _utils.normalizeData(featureVectors, False, spatialWeight=spatialWeight)

    print("Clustering...")
    labels = _utils.clusterFeatureVectors(featureVectors, k_clusters)
    _utils.printClassifiedImage(labels, k_clusters, img, o, greyOutput)

def main():
    a = "/Users/Christuuu/Desktop/TASK3/in.png"
    o = "/Users/Christuuu/Desktop/TASK3/out.png"
    k = 16
    gk = 7
    M = 7
    spw=1
    gamma=1
    psi=0
    vt=0.0001
    fi=100
    R=0.95
    siw=0.5
    c=True
    i=False
    sigma=1
    runGabor(a,o,k,gk,M,spw,gamma,sigma,psi,vt,fi,R,siw,c,i)

if __name__ == "__main__":
    main()
