import pandas as pd
import re
from datetime import datetime

class NewsClusters:
    
    def __init__(self, data, k =3):
        self.data = data
        self.k   = k
        ##Start with random centres
        self.centroids  = data.sample(k, random_state = datetime.now().second)
        self.previousCentroids = []
        self.data['cluster'] =  -1
    
    def jaccard_distance(self, a, b):
        union = len(a.union(b))
        intersection = len(a.intersection(b))
        return 1 - (intersection/union)
    
    def getTotalDistance(self, point, data):
        sum = 0
        for index, d in data.iterrows():
            sum += self.jaccard_distance(point.processedTweet, d.processedTweet)
        return sum
    
    ##Step 1: Assign Clusters to Tweets
    
    def assign_clusters(self, df, centroids):
        for index, data in df.iterrows():
            distance = {}
            tweet = data.processedTweet
            ##Calculate distance of a tweet to each of the centroids
            for i in range(0, self.k):
                distance[i] = self.jaccard_distance(centroids.iloc[i].processedTweet, tweet)
            ##Assign the tweet to the cluster which has minimum distance
            df.at[index ,'cluster'] = min(distance, key=lambda k: distance[k])

    ##Step 2: Update centroids
    def update_centroids(self, df):
        group = df.groupby('cluster')
        newCentroids = pd.DataFrame()
        for i in range(0, self.k):
            g = group.get_group(i)
            ##Calculate total distance of one tweet to all other tweets in the cluster
            g['distance'] = g.apply(lambda x: self.getTotalDistance(x, g), axis=1)
                ##Set the tweet that has the lowest net distance as Centroid
            newCentroids = newCentroids.append(g.loc[[g['distance'].idxmin()]])
            return newCentroids
    
    def calulateSSE(self):
        SSE = 0
        for i, d in self.data.iterrows():
            SSE += (self.jaccard_distance(d.processedTweet, self.centroids.iloc[d.cluster].processedTweet))**2
        return SSE

    def fit(self):
        iteration = 1
        print("Initial Centres:")
        print(self.centroids)
        while not self.centroids.equals(self.previousCentroids):
                print("Iteration #", iteration)
                iteration += 1
                self.assign_clusters(self.data, self.centroids)
                print("Clusters Assigned" + "\n" + "Updating centroids")
                centroids_new = self.update_centroids(self.data)
                print("New Centres:")
                print(centroids_new)
                self.previousCentroids = self.centroids
                self.centroids = centroids_new

    def getClusters(self):
        return self.data.groupby('cluster')
        
    def report(self):
            SSE = self.calulateSSE()
            count = self.getClusters()['tweet'].count()
            for i in range(0, self.k):
                print('Cluster ', i+1, ' ', count[i], ' tweets')
            print('SSE: ', SSE)

def preprocess(DataFrame):
    # Remove columns tweetID and Timestamp
    DataFrame.drop(columns=['tweetId', 'dateTime'], inplace=True)
    # Remove '@' , '#' , URLs and convert to lowercase
    DataFrame['tweets'] = DataFrame['tweet'].apply(lambda s: [k.strip('#') for k in s.lower().split()])
    DataFrame['tweets'] = DataFrame['tweets'].apply(lambda s: [k.strip('@') for k in s])
    DataFrame['tweets'] = DataFrame['tweets'].apply(lambda s: list(filter((lambda x: not re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)), s)))
    DataFrame['processedTweet'] = DataFrame['tweets'].apply(lambda s: set(s))
