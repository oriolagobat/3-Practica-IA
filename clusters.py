from typing import Tuple, List
from math import sqrt
import random
import sys

def readfile(filename: str) -> Tuple[List, List, List]:
    """
    Parses input file.
    """
    headers = None
    row_names = []
    data = []

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
    """
    Computes normalized euclidean distance between two vectors.
    """
    distance = sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return 1 / (1 + distance)


def euclidean_squared(v1, v2):
    """
    Computes squared euclidean distance between two vectors.
    """
    return euclidean(v1, v2) ** 2


def pearson(v1, v2):
    """
    Computes pearson distance between two vectors.
    """
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


# ......... K-MEANS ..........
class KMeans:
    """
    Represents k-means algorithm.
    """
    def __init__(self, rows, distance=euclidean_squared, k=4):
        self.rows = rows
        self.distance = distance
        self.k = k
        self.clusters = None

    def find_best_startconfig(self, iterations=10, bestresult=None):
        """
        Applies KMeans with a restarting policy and stores the best result based on
        the total distance of each item to it's closest centroid.
        """
        for _ in range(iterations):
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
        for _ in range(iterations):
            #print("Iteració %i" % t)
            #print(centroids[0][0])
            distances, matches = self._find_closest_centroids(centroids)

            # If the results haven't changed, stop
            if matches == lastmatches:
                break
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
        for rowid, row in enumerate(self.rows):

            # Set centroid 0 as closest by default
            bestmatch = 0
            bestdistance = self.distance(centroids[0], row)

            # Find the closest centroid
            for centroid_id, centroid in enumerate(centroids):
                distance = self.distance(centroid,row)
                if distance > self.distance(centroids[bestmatch],row):
                    bestmatch=centroid_id
                    bestdistance=distance

            # Store results
            bestdistances[rowid] = bestdistance
            bestmatches[bestmatch].append(rowid)

        return (bestdistances, bestmatches)


    def _move_centroids(self, centroids, matches):
        """
        Updates each centroid's position.
        """
        # For each centroid
        for centroid_id, _ in enumerate(centroids):
        #for i in range(len(centroids)):
            avgs = [0.0] * len(self.rows[0])

            # That has at least one item
            if len(matches[centroid_id]) > 0:

                # For each item it has
                for rowid in matches[centroid_id]:

                    # Add the value of it's attributes to the subtotal
                    for attrid in range(len(self.rows[rowid])):
                        avgs[attrid] += self.rows[rowid][attrid]

                # For each attribute get the mean value
                for j in range(len(avgs)):
                    avgs[j]/=len(matches[centroid_id])

                    # Move centroid
                    centroids[centroid_id]=avgs


def test_different_kvalues(data, begin, end, incr, restarts):
    """
    Given dataset and a range of k values and a restart policy value, gives
    back the kmeans total distance for each value. Used for analisis purposes.
    """
    totaldistances  = []
    for i in range(begin, end, incr):
        kmeans = KMeans(data, k=i)
        _, totaldistance = kmeans.find_best_startconfig(iterations=restarts)
        totaldistances.append(totaldistance)
    return totaldistances


def main():
    # Parse input
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "blogdata.txt"
    _, _, data = readfile(filename)

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
    centroids, totaldistance = kmeans.find_best_startconfig(iterations=restarts)
    print("[Apartat 2] (%i Restarts) Distància total als centroides: %2.3f\n"
        % (restarts, totaldistance))

    # APARTAT 3 (t10)
    result = test_different_kvalues(data, 1, 5, 1, restarts)
    for i in range(len(result)):
        print("[Apartat 3] (%i Clusters - %i Restarts) Distància total als centroides: %2.3f"
            % (i + 1, restarts, result[i]))


if __name__ == "__main__":
    main()
