from typing import Tuple, List
from math import sqrt
import random
import sys

def readfile(filename: str) -> Tuple[List, List, List]:
    headers = None
    row_names = list()
    data = list()

    with open(filename) as file_:
        for line in file_:
            values = line.strip().split("\t")
            if headers is None:
                headers = values[1:]
            else:
                row_names.append(values[0])
                data.append([float(x) for x in values[1:]])
    return row_names, headers, data


# .........DISTANCES........
# They are normalized between 0 and 1, where 1 means two vectors are identical
def euclidean(v1, v2):
    distance = sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return 1 / (1 + distance)


def euclidean_squared(v1, v2):
    return euclidean(v1, v2) ** 2


def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
    # Sums of squares
    sum1sq = sum([v ** 2 for v in v1])
    sum2sq = sum([v ** 2 for v in v2])
    # Sum of the products
    products = sum([a * b for (a, b) in zip(v1, v2)])
    # Calculate r (Pearson score)
    num = products - (sum1 * sum2 / len(v1))
    den = sqrt((sum1sq - sum1 ** 2 / len(v1)) * (sum2sq - sum2 ** 2 / len(v1)))
    if den == 0:
        return 0
    return 1 - num / den


# ........HIERARCHICAL........
class BiCluster:
    def __init__(self, vec, left=None, right=None, dist=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = dist


def hcluster(rows, distance=pearson):
    distances = {}  # Cache of distance calculations
    currentclustid = -1  # Non original clusters have negative id

    # Clusters are initially just the rows
    clust = [BiCluster(row, id=i) for (i, row) in enumerate(rows)]

    """
    while ...:  # Termination criterion
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                distances[(clust[i].id, clust[j].id)] = ...

            # update closest and lowestpair if needed
            ...
        # Calculate the average vector of the two clusters
        mergevec = ...

        # Create the new cluster
        new_cluster = BiCluster(...)

        # Update the clusters
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(new_cluster)
    """

    return clust[0]


def printclust(clust: BiCluster, labels=None, n=0):
    # indent to make a hierarchy layout
    indent = " " * n
    if clust.id < 0:
        # Negative means it is a branch
        print(f"{indent}-")
    else:
        # Positive id means that it is a point in the dataset
        if labels == None:
            print(f"{indent}{clust.id}")
        else:
            print(f"{indent}{labels[clust.id]}")
    # Print the right and left branches
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n + P1)



# ......... K-MEANS ..........
class KMeans:
    def __init__(self, rows, distance=euclidean_squared, k=4):
        self.rows = rows
        self.distance = distance
        self.k = k
        self.clusters = None

    def find_best_startconfig(self, data, iterations=10, bestresult=None):
        """
        Applies KMeans with a restarting policy and stores the best result based on
        the total distance of each item to it's closest centroid.
        """
        for i in range(iterations):
            # Find clusters
            initial_centroids = self.init_centroids()
            centroids, totaldistance = self.find_clusters(initial_centroids)

            # If it's the first/best result, store it
            if bestresult is None or totaldistance > bestresult[1]:
                bestresult = (centroids, totaldistance)

        return bestresult

    def init_centroids(self):
        """
        Initializes k centroids with random values.
        """
        # Determine the minimum and maximum values for each attribute
        ranges = [(min([row[i] for row in self.rows]),
        max([row[i] for row in self.rows])) for i in range(len(self.rows[0]))]

        # Create k randomly placed centroids within the ranges found
        centroids = [[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]
        for i in range(len(self.rows[0]))] for j in range(self.k)]

        return centroids


    def find_clusters(self, centroids, iterations=100, lastmatches=None):
        """
        Applies given iterations of cluster finding.
        """
        # Update clusters on each iteration
        for t in range(iterations):
            #print("Iteració %i" % t)
            #print(centroids[0][0])
            distances, matches = self._find_closest_centroids(centroids)

            # If the results haven't changed, stop
            if matches == lastmatches: break
            lastmatches = matches

            # Update centroid positions
            self._move_centroids(centroids, matches)

        self.clusters = matches
        return (centroids, sum(distances))


    def _find_closest_centroids(self, centroids):
        """
        Finds closest centroids to each item.
        """
        # Initialize lists
        bestmatches = [[] for i in range(len(centroids))]
        bestdistances = [0 for i in range(len(self.rows))]

        # For each row
        for j in range(len(self.rows)):

            # Set centroid 0 as closest by default
            row = self.rows[j]
            bestmatch = 0
            bestdistance = self.distance(centroids[0], row)

            # Find the closest centroid
            for i in range(len(centroids)):
                d = self.distance(centroids[i],row)
                # print("Distancia a bestmatch: %i Distancia a K%i: %i" % (distance(clusters[bestmatch],row), i, d), file=sys.stderr)
                if d > self.distance(centroids[bestmatch],row): 
                    bestmatch=i
                    bestdistance=d
            
            # Store results
            bestdistances[j] = bestdistance
            bestmatches[bestmatch].append(j)
        
        return (bestdistances, bestmatches)

    
    def _move_centroids(self, centroids, matches):
        """
        Updates each centroid's position.
        """
        # For each centroid
        for i in range(len(centroids)):
            avgs = [0.0] * len(self.rows[0])

            # That has at least one item
            if len(matches[i]) > 0:

                # For each item it has
                for rowid in matches[i]:

                    # Add the value of it's attributes to the subtotal
                    for attrid in range(len(self.rows[rowid])):
                        avgs[attrid] += self.rows[rowid][attrid]

                # For each attribute get the mean value
                for j in range(len(avgs)):
                    avgs[j]/=len(matches[i])

                    # Move centroid
                    centroids[i]=avgs
    

def test_different_kvalues(data, begin, end, incr, restarts, totaldistances=[]):
    """
    Given dataset and a range of k values and a restart policy value, gives 
    back the kmeans total distance for each value. Used for analisis purposes.
    """
    for i in range(begin, end, incr):
        kmeans = KMeans(data, k=i)
        _, totaldistance = kmeans.find_best_startconfig(data, iterations=restarts)
        totaldistances.append(totaldistance)
    return totaldistances


def main():
    # Parse input
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "blogdata.txt"
    row_names, headers, data = readfile(filename)

    # APARTAT 1 (t9)
    kmeans = KMeans(data)
    initial_centroids = kmeans.init_centroids()
    centroids, totaldistance = kmeans.find_clusters(initial_centroids)
    print("[Apartat 1] Distància total als centroides: %2.3f\n" % totaldistance)

    # To check the resulting clusters
    # print(kmeans.clusters)

    # APARTAT 2 (t11)
    restarts = 10
    kmeans = KMeans(data)
    centroids, totaldistance = kmeans.find_best_startconfig(data, iterations=restarts)
    print("[Apartat 2] (%i Restarts) Distància total als centroides: %2.3f\n" % (restarts, totaldistance))

    # APARTAT 3 (t10)
    result = test_different_kvalues(data, 1, 5, 1, restarts)
    for i in range(len(result)):
        print("[Apartat 3] (%i Clusters - %i Restarts) Distància total als centroides: %2.3f" % (i + 1, restarts, result[i]))


if __name__ == "__main__":
    main()