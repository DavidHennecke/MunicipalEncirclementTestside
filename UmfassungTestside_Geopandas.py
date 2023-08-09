# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:44:09 2023

@author: hennecke
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:07:58 2023

@author: Hennecke
"""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import Point, Polygon, LineString, distance
import math
from tqdm import tqdm
import rasterio
from rasterstats import point_query
import time
import concurrent.futures
from geopandas.tools import sjoin
picNum = 0;
print("Calculate Enclosure:")

#PARAMETER
SegmentDistance = 200
bufferRule = 3500

#DATA PATHS
totalStockPath = r"Testdaten/stock_32633.shp"
locationsPath = r"Testdaten/Blankenhagen_32633.shp"
dsmPath = r"Testdaten/SRTM_D_32633.tif"



#load DSM-Raster
dsm = rasterio.open(dsmPath)
dsmData = dsm.read(1)
transform = dsm.transform

#load Vectordata
totalStock = gpd.read_file(totalStockPath)
totalStock = totalStock.explode(ignore_index=True)
locations = gpd.read_file(locationsPath)
locations = locations.explode(ignore_index=True)


#progressBar Init
pbar = tqdm(total=len(locations))

#init Results
allowedAreasFinalList = []
restrictedAreasFinalList = []
forbiddenAreasFinalList = []
   
print("Calculate Enclosure:")

#Calculates points along a polygon boundary
def polygonToPoints(locationGeom, SegmentDistance):
    line = locationGeom.boundary
    points = []
    for i in np.arange(0, line.length, SegmentDistance):
        val = line.interpolate(i)
        points.append([val.x, val.y])
    points = np.array(points)
    return points

def lineToPoints(line, SegmentDistance):
    points = []
    for i in np.arange(0, line.length, SegmentDistance):
        val = line.interpolate(i)
        points.append([val.x, val.y])
    return points

def checkLineOfSight (observerGeom, targetGeom, dsmData, transform, hubHeight, observerHeight):
    sight = True
    #get Line of Sight and Points on Line
    segmentSize = transform[0]
    LoS = LineString([observerGeom, targetGeom])
    LoSPoints = lineToPoints(LoS, segmentSize)
    #get Heights and Distance
    
    targetHeight = point_query(targetGeom.wkt,dsmData, affine=transform, nodata=0)
    distanceBetweenObserverTarget = distance(observerGeom, targetGeom)
    
    #interpolation between two points x = distance; y= height
    distances = [0,distanceBetweenObserverTarget]
    heights = [observerHeight[0] + 1.65, targetHeight[0]+hubHeight] #1.65 m average size of a human being
    
    if targetHeight == None or observerHeight == None:
        return sight
    baseHeight = observerHeight[0]
    for point in LoSPoints:
        pointRealHeight = point_query(Point(point).wkt,dsmData, affine=transform, nodata=0)[0]
        if pointRealHeight >= baseHeight + 10 or pointRealHeight <= baseHeight -10:
            height = np.interp(segmentSize, distances, heights)
            segmentSize = segmentSize + segmentSize
            
            if pointRealHeight > height:
                sight = False
                return sight
            else:
                sight = True
    return sight

#Calculates Turbines within a location-buffer
def getStock(locationGeom, stock):
    buffer = locationGeom.buffer(bufferRule)
    gdfBuffer = gpd.GeoDataFrame(geometry=gpd.GeoSeries(buffer), crs = stock.crs)
    # checks each turbine whether they are inside the buffer or not
    stockInside = sjoin(stock,gdfBuffer)
    stockInside.drop('index_right', axis=1, inplace=True)
    return stockInside


def calculateAzimuth(observer, turbines):
    X = turbines[:,0] - observer[:,0]
    Y = turbines[:,1] - observer[:,1]
    bearingList = np.arctan2(X,Y)*(180/np.pi)
    bearingList[bearingList < 0] += 360
    return bearingList   


def plot(observer, turbines, observerList, area1, area2, area3, area4):
    global picNum
    #plots the result
    fig, axs = plt.subplots()
    axs.set_aspect('equal')
    xs = turbines[:,0]  
    ys = turbines[:,1]
    xo = observer[:,0]  
    yo = observer[:,1] 
    
    xlArea1 = area1[:,0]  
    ylArea1 = area1[:,1]
    axs.fill(xlArea1, ylArea1, alpha=0.7, facecolor="seagreen", edgecolor='darkgreen') #Plot allowedAreas
    
    xlArea4 = area4[:,0]  
    ylArea4 = area4[:,1]
    axs.fill(xlArea4, ylArea4, alpha=0.7, facecolor="cornflowerblue", edgecolor='royalblue') #Plot windfarmAreas
    
    xlArea2 = area2[:,0]  
    ylArea2 = area2[:,1]
    axs.fill(xlArea2, ylArea2, alpha=0.7, facecolor="orange", edgecolor='darkorange') #Plot restrictedAreas
    
    xlArea3 = area3[:,0]  
    ylArea3 = area3[:,1]
    axs.fill(xlArea3, ylArea3, alpha=0.7, facecolor="indianred", edgecolor='darkred') #Plot forbiddenAreas
    
    xl = observerList[:,0]  
    yl = observerList[:,1]
    axs.fill(xl, yl, alpha=1, facecolor="slategrey", edgecolor="black") #Plot observerList
    
    #axs.plot(xo, yo, "o") #Plot observer
    axs.plot(xs, ys, ".", color="black", alpha=1) #Plot turbines
    axs.set_xlim([320000, 333000])
    axs.set_ylim([5995500, 6011500])
    axs.set_xticks([320000, 323250, 326500, 329750, 333000])
    axs.set_yticks([5995500, 5999750, 6003500, 6007250, 6011500])
    plt.xlabel("x", 
           family='serif', 
           color='k', 
           weight='normal', 
           size = 12,
           labelpad = 6)
    plt.ylabel("y", 
           family='serif', 
           color='k', 
           weight='normal', 
           size = 12,
           labelpad = 6)
    plt.savefig('Ergebnisse/Animation/test_{}.svg'.format(picNum))
    picNum+=1
    plt.show()

def plotPolygon(observer, turbines, observerList, area1, area2):
    global picNum
    #plots the result
    fig, axs = plt.subplots()
    axs.set_aspect('equal')
    xs = turbines[:,0]  
    ys = turbines[:,1]
    xo = observer[:,0]  
    yo = observer[:,1] 
    
    if len(area1)>0:
        for area in area1:
            xlArea1, ylArea1 = area.exterior.xy
            axs.plot(xlArea1, ylArea1, color="orange", linestyle='dashed')
        
    if len(area2)>0:
        for area in area2:
            xlArea2, ylArea2 = area.exterior.xy
            axs.fill(xlArea2, ylArea2, alpha=0.8, facecolor="indianred", edgecolor='orangered') #Plot forbiddenAreas

    xl = observerList[:,0]  
    yl = observerList[:,1]
    axs.fill(xl, yl, alpha=1, facecolor="slategrey") #Plot observerList
    
    axs.plot(xo, yo, "o", color="red") #Plot observer
    axs.plot(xs, ys, ".", color="black", alpha=1) #Plot turbines
    axs.set_xlim([320000, 333000])
    axs.set_ylim([5995500, 6011500])
    axs.set_xticks([320000, 323250, 326500, 329750, 333000])
    axs.set_yticks([5995500, 5999750, 6003500, 6007250, 6011500])
    plt.xlabel("x", 
           family='serif', 
           color='k', 
           weight='normal', 
           size = 12,
           labelpad = 6)
    plt.ylabel("y", 
           family='serif', 
           color='k', 
           weight='normal', 
           size = 12,
           labelpad = 6)
    plt.savefig('Ergebnisse/Animation/test_{}.svg'.format(picNum))
    picNum+=1
    plt.show()

def generateAngleGroups(bearingList, observer):
    bearingList = np.where(bearingList > 180, bearingList-360, bearingList)
    bearingList = np.sort(bearingList)
    
    groups = []
    for angle in bearingList:
        if len(groups) == 0:
            groups.append([angle])
        else:
            foundGroup = False
            for i, group in enumerate(groups):
                if max(group) >= angle - 60:
                    group.append(angle)
                    foundGroup = True
            if foundGroup == False:
                groups.append([angle])
    return groups   


def generateAreas (areaAngles, observer):
    left = areaAngles[0]
    minG = areaAngles[1]
    maxG = areaAngles[2]
    right = areaAngles[3]
    rings = []
    ring = []
    ring.append((observer[:,0][0], observer[:,1][0]))
    while left <= minG:
        calcAngle = 0
        if left < 0:
            calcAngle = left + 360
        else:
            calcAngle = left
        xT = observer[:,0][0] + math.sin(math.radians(calcAngle))*bufferRule
        yT = observer[:,1][0] + math.cos(math.radians(calcAngle))*bufferRule
        ring.append((xT, yT))
        left = left + 1
    ring.append((observer[:,0][0], observer[:,1][0]))
    rings.append(Polygon(ring))
    ring = []
    ring.append((observer[:,0][0], observer[:,1][0]))
    while right >= maxG:
        calcAngle = 0
        if right < 0:
            calcAngle = right + 360
        else:
            calcAngle = right
        xT = observer[:,0][0] + math.sin(math.radians(calcAngle))*bufferRule
        yT = observer[:,1][0] + math.cos(math.radians(calcAngle))*bufferRule
        ring.append((xT, yT))
        right = right -1
    ring.append((observer[:,0][0], observer[:,1][0]))
    rings.append(Polygon(ring))
    return rings

def evaluateGroups(groups, observer):
    #Init Areas
    restrictedAreasGroup = []
    forbiddenAreasGroup = []

    # Iteration through groups list
    for i, group in enumerate(groups):
        minG = min(group)
        maxG = max(group)
        angleRange = maxG-minG
        left = 0
        right = 0
        areaAngles = []
        if angleRange >= 120:
            left = minG - 60
            right = maxG + 60
            areaAngles = np.array([left, minG, maxG, right])
            forbiddenAreasGroup.extend(generateAreas (areaAngles, observer))
        
        else:
            left = minG - (120-angleRange)
            right = maxG + (120-angleRange)
            areaAngles = np.array([left, minG, maxG, right])
            restrictedAreasGroup.extend(generateAreas (areaAngles, observer))
        
    return restrictedAreasGroup, forbiddenAreasGroup  


def calculateLocation(location):
    
    #Init Outputs
    restrictedAreasList = []
    forbiddenAreasList = []
    
    global windfarmAreasListFinal
    global restrictedAreasListFinal
    global forbiddenAreasListFinal
    global start_mem
    global delta_mem
    global max_memory
    
    #Get location geometry and calculate ObserverPoins
    locationGeom = location[1].geometry
    observerList = polygonToPoints(locationGeom, SegmentDistance)
    
    #Get affected turbines by location
    locationStock = getStock(locationGeom, totalStock)
    
    
    #Calculates enclosure for each observer
    #pbar = tqdm(total=len(observerList))
    for observer in observerList:
        #Get affected turbines by observer
        observerGeom = Point(observer[0], observer[1])
        observerStock = getStock(observerGeom, locationStock)
        
        #check Line of Sight 
        observerHeight = point_query(observerGeom.wkt,dsmData, affine=transform, nodata=0)
        for targetId, target in observerStock.iterrows():
            targetGeom = target.geometry
            #hubHeight = float(target.Nabenhoehe)
            hubHeight = float(target.Height)
            sight = checkLineOfSight (observerGeom, targetGeom, dsmData, transform, hubHeight, observerHeight)
            if sight == False:
                observerStock = observerStock.drop(target.name)
            
        #Calculation continues if turbines are nearby
        if observerStock.size != 0:
            #change observer to a list
            observer = np.vstack((observer[0], observer[1])).T
            #observerStock to numpy array
            observerStockNP = np.array([ np.array((geom.xy[0][0], geom.xy[1][0])) for geom in observerStock.geometry ])
            #Calculates angles to turbines of observerStock
            bearingList = calculateAzimuth(observer, observerStockNP)
            #Find groups of turbines by angle
            groups = generateAngleGroups(bearingList, observer)
            #Evaluate turbine groups 
            if len(groups)>0:
                restrictedArea, forbiddenArea = evaluateGroups(groups, observer)
                restrictedAreasList.extend(restrictedArea)
                forbiddenAreasList.extend(forbiddenArea)
            
            #pbar.update(1)
            #plotPolygon(observer, observerStockNP, observerList, restrictedArea, forbiddenArea)
    bufferAllowedArea = locationGeom.buffer(bufferRule)
    allowedArea = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bufferAllowedArea), crs = totalStock.crs)
    restrictedArea = gpd.GeoDataFrame(geometry=gpd.GeoSeries(restrictedAreasList), crs = totalStock.crs)
    forbiddenArea = gpd.GeoDataFrame(geometry=gpd.GeoSeries(forbiddenAreasList), crs = totalStock.crs)    

    allowedAreasFinalList.append(bufferAllowedArea)
    restrictedAreasFinalList.append(restrictedArea.geometry.unary_union)
    forbiddenAreasFinalList.append(forbiddenArea.geometry.unary_union)
    
    base = allowedArea.plot(color='green')
    restrictedArea.plot(ax=base, color= 'orange')
    forbiddenArea.plot(ax=base, color= 'red')
    gpd.GeoSeries(locationGeom).plot(ax=base, color= 'grey')
    plt.plot()
    plt.show()
    
    #pbar.close()  
    pbar.update(1)
    

# MAIN
def main():
    global allowedAreasFinalList
    global restrictedAreasFinalList
    global forbiddenAreasFinalList
    #get calculation time
    st = time.time()
    #Calculates enclosure for each location
    # with concurrent.futures.ThreadPoolExecutor() as executer:
    #     executer.map(calculateLocation, locations.iterrows())
        
    
    for location in locations.iterrows():
         calculateLocation(location)
    
    
    allowedAreasFinal = gpd.GeoDataFrame(geometry=gpd.GeoSeries(allowedAreasFinalList), crs = totalStock.crs)
    restrictedAreasFinal = gpd.GeoDataFrame(geometry=gpd.GeoSeries(restrictedAreasFinalList), crs = totalStock.crs)
    forbiddenAreasFinal = gpd.GeoDataFrame(geometry=gpd.GeoSeries(forbiddenAreasFinalList), crs = totalStock.crs)
    
    allowedAreasFinalList = []
    restrictedAreasFinalList = []
    forbiddenAreasFinalList = []
    
    allowedAreasExport = gpd.GeoDataFrame(geometry=gpd.GeoSeries([allowedAreasFinal.geometry.unary_union]), crs = totalStock.crs)
    restrictedAreasExport  = gpd.GeoDataFrame(geometry=gpd.GeoSeries([restrictedAreasFinal.geometry.unary_union]), crs = totalStock.crs)
    forbiddenAreasExport = gpd.GeoDataFrame(geometry=gpd.GeoSeries([forbiddenAreasFinal.geometry.unary_union]), crs = totalStock.crs)
    
    # allowedAreasExport.to_file('Ergebnisse/allowedAreas.shp')  
    # restrictedAreasExport.to_file('Ergebnisse/restrictedAreas.shp')  
    # forbiddenAreasExport.to_file('Ergebnisse/forbiddenAreas.shp')  
    
    pbar.close()  

    
    et = time.time()   
    elapsedTime = et - st
    print ("Elapsed time: "+ str(elapsedTime))

        
        
if __name__ == "__main__":
    main()
    



