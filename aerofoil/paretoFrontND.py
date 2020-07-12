# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:25:36 2020

@author: Clym Stock-Williams

Collates functions which analyse ND archives from multi-objective optimisation.
N.B. All functions assume minimisation.
"""

from deap.tools._hypervolume import hv
import numpy as np
import scipy.spatial.distance as dist

def isDominatedBy(d1PointTest, d1PointCompare):
    """Tests if a point is dominated by another.
    N.B. works in any number of dimensions.
    
    d1PointTest     -- a 2-element list specifying the location to test domination for
    d1PointCompare  -- a 2-element list specifying the reference location to compare with
    
    Returns a Boolean (true if the test point is dominated)
    """
    return all(d1PointCompare <= d1PointTest) and any(d1PointCompare < d1PointTest)

def getNonDominatedFront(d2Points):
    """Returns the first Pareto Front by removing points which are dominated.
    N.B. works in any number of dimensions.
    
    d2Points -- a 2D array of locations from an archive (each row is a location)
    
    Returns a 2D numpy array of non-dominated locations (each row is a location)
    """
    b1OnFront = np.array([False for i in range(d2Points.shape[0])])
    # Check each point against every other point for domination
    for i in range(d2Points.shape[0]):
        bDominated = False
        for j in range(d2Points.shape[0]):
            if i == j:
                continue
            if isDominatedBy(d2Points[i, :], d2Points[j, :]):
                bDominated = True
                break
        if not bDominated:
            b1OnFront[i] = True
    # Filter only those which are non-dominated
    return d2Points[b1OnFront, :], b1OnFront

def fastNonDominatedSort(d2Points):
    """ Returns the Pareto Front number for each of a list of points.
    
    d2Points -- a 2D array of locations from an archive (each row is a location)
    
    Returns a list of integers representing the PF (starting at 1)
    """
    dominationSets = []
    numsDominating = np.ones((d2Points.shape[0], 1)) * -1
    ranks = np.ones((d2Points.shape[0], 1)) * -1
    temp_this_front = []
    for p in range(d2Points.shape[0]):
        dominationSets.append([])
        numsDominating[p] = 0
        for q in range(d2Points.shape[0]):
            if isDominatedBy(d2Points[q, :], d2Points[p, :]):
                # p dominates q
                dominationSets[p].append(q)
            elif isDominatedBy(d2Points[p, :], d2Points[q, :]):
                # q dominates p
                numsDominating[p] += 1
        if numsDominating[p] == 0:
            ranks[p] = 1
            temp_this_front.append(p)
    iFront = 1
    while len(temp_this_front) > 0:
        temp_next_front = []
        for p in temp_this_front:
            for q in dominationSets[p]:
                numsDominating[q] -= 1
                if numsDominating[q] == 0:
                    ranks[q] = iFront + 1
                    temp_next_front.append(q)
        iFront += 1
        temp_this_front = temp_next_front
    return ranks

def calculateIGD(d2TruePf, d2NonDomSet):
    """ Returns the inverted generational distance (IGD) between 
    a list of points representing the true Pareto Front and
    the non-dominated set found from an optimisation.
    
    d2TruePf    -- a 2D array of locations from a Pareto Front (each row is a location)
    d2NonDomSet -- a 2D array of locations from an archive (each row is a location)
    
    Returns a float representing the IGD
    """
    igds = []
    for p in range(d2TruePf.shape[0]):
        d1TruePoint = d2TruePf[p,:]
        dists = []
        for i in range(d2NonDomSet.shape[0]):
            d1TestPoint = d2NonDomSet[i,:]
            dists.append(dist.euclidean(d1TestPoint, d1TruePoint))
        igds.append(min(dists)**2.0)
    return np.sqrt(np.sum(np.array(igds))) / len(igds)

def calculateHypervolume(d2Points, d1Reference):
    """Returns the Hypervolume Indicator (total area contained by the Pareto Front).
    
    d2Points    -- a 2D array of locations (each row is a location)
    d1Reference -- a D-element list specifying the nadir point (e.g. [5, 5, 5, 5])
    
    Returns a float for the Hypervolume Indicator value
    """
    return hv.hypervolume(d2Points, d1Reference)

def calculateHypervolumeContributions(d2Points, d1Reference):
    """Returns the Hypervolume Contribution for each location
    (additional area contributed to the Pareto Front).
    
    d2Points    -- a 2D array of locations from an archive (each row is a location)
    d1Reference -- a D-element list specifying the nadir point (e.g. [5, 5, 5, 5])
    
    Returns a 2D numpy array column with the Hypervolume Contributions of each location
    (0 if the location is not on the Pareto Front)
    """
    d1Hvc = np.zeros((d2Points.shape[0], 1))
    # Get the hypervolume of the non-dominated front with all points
    dTotalHyperVolume = calculateHypervolume(d2Points, d1Reference)
    # Get the hypervolume of the non-dominated front without each point
    for i in range(d2Points.shape[0]):
        d2PointsExcl = np.delete(d2Points, i, 0)
        dHyperVolumeExcl = calculateHypervolume(d2PointsExcl, d1Reference)
        d1Hvc[i, 0] = dTotalHyperVolume - dHyperVolumeExcl
    return d1Hvc

def calculateNegativeHypervolumeContributions(d2Points):
    """Returns the Negative Hypervolume Contribution for each location
    (area between the location and the Pareto Front).
    
    d2Points -- a 2D array of locations from an archive (each row is a location)
    
    Returns a 2D numpy array column with the Negative Hypervolume Contributions of each location
    (0 if the location is on the Pareto Front)
    """
    d1HvcNeg = np.zeros((d2Points.shape[0], 1))
    # Get the hypervolume of the non-dominated front with each point as reference
    for i in range(d2Points.shape[0]):
        d1HvcNeg[i, 0] = calculateHypervolume(d2Points, d2Points[i, :])
    return d1HvcNeg

def calculatePotentialHvcsGivenFront(d2NewPoints, d2CurrentPoints, d1Reference):
    """Returns the Hypervolume Contribution for potential new locations,
    with respect to an established Pareto Front.
    
    d2NewPoints     -- a 2D array of potential new locations (each row is a location)
    d2CurrentPoints -- a 2D array of non-dominated locations (each row is a location)
    d1Reference     -- a D-element list specifying the nadir point (e.g. [5, 5, 5, 5])
    
    Returns a 2D numpy array column with the Hypervolume Contribution for each new location,
    if it were added to the existing archive.
    """
    d1Hvc = np.zeros((d2NewPoints.shape[0], 1))
    # Get the hypervolume of the current non-dominated front
    # N.B. This is why we don't just use calculateHypervolumeContributions().
    dTotalHyperVolume = calculateHypervolume(d2CurrentPoints, d1Reference)
    # Get the hypervolume of the non-dominated front including each new point
    for i in range(d2NewPoints.shape[0]):
        d1TestPoint = d2NewPoints[i, :]
        dHyperVolumeIncl = calculateHypervolume(
            np.vstack((d2CurrentPoints, d1TestPoint)),
            d1Reference)
        d1Hvc[i, 0] = dHyperVolumeIncl - dTotalHyperVolume
    return d1Hvc

def calculatePotentialNegativeHvcsGivenFront(d2NewPoints, d2CurrentPoints):
    """Returns the Negative Hypervolume Contribution for potential new locations,
    with respect to an established Pareto Front.
    
    d2NewPoints     -- a 2D array of potential new locations (each row is a location)
    d2CurrentPoints -- a 2D array of non-dominated locations (each row is a location)
    
    Returns a 2D numpy array column with the Negative Hypervolume Contributions of each location,
    if it were added to the existing archive.
    """
    # Calculate negative hypervolume contribution
    d1HvcNeg = np.zeros((d2NewPoints.shape[0], 1))
    # Get the hypervolume of the non-dominated front with each new point as reference
    for i in range(d2NewPoints.shape[0]):
        d1TestPoint = d2NewPoints[i, :]
        d1HvcNeg[i, 0] = calculateHypervolume(
            np.vstack((d2CurrentPoints, d1TestPoint)),
            d1TestPoint)
    return d1HvcNeg

# def calculateProbabilityOfImprovementMC(d1Mean, d1Std, d2CurrentPoints, iNumSamples=1000):
#     """Returns the Probability of Improvement for a new location with
#     independent Gaussian uncertainty, given an existing exact Pareto Front.
#      N.B. works in any number of dimensions.
    
#     d1Mean          -- a 2-element list specifying the mean location of the new point
#     d1Std           -- a 2-element list specifying the standard deviation of uncertainty 
#                         on the location of the new point
#     d2CurrentPoints -- a 2D array of non-dominated locations (each row is a location)
#     iNumSamples     -- the number of Monte Carlo samples to draw from the location's 
#                         Normal distribution (default = 1000)
    
#     Returns a float for the Probability of Improvement of the new location over the existing Pareto Front
#     """
#     d2Samples = np.array(np.random.normal(d1Mean[0], d1Std[0], iNumSamples))
#     for i in range(1, len(d1Mean)):
#         d2Samples = np.vstack([d2Samples,
#                                np.random.normal(d1Mean[i], d1Std[i], iNumSamples)])
#     d2Samples = d2Samples.T
#     # Determine if the each new sample point joins the Pareto Front
#     improvements = 0
#     for i in range(d2Samples.shape[0]):
#         d1TestPoint = d2Samples[i, :]
#         d2NewFront = getNonDominatedFront(np.vstack((d2Front, d1TestPoint)))
#         # We also count perfect replacements, rare as they might be, as improvements!
#         if any(d2NewFront[:, 0] == d1TestPoint[0]) and any(d2NewFront[:, 1] == d1TestPoint[1]):
#             improvements = improvements + 1
#     # Return the improvement ratio
#     return improvements / iNumSamples

def calculateHypIs(d2Points, d1Reference):
    """Returns the Hypervolume Improvements for each location
    (additional area contributed to the *local* Pareto Front).
    
    d2Points    -- a 2D array of locations from an archive (each row is a location)
    d1Reference -- a D-element list specifying the nadir point (e.g. [5, 5, 5, 5])
    
    Returns a 2D numpy array column with the Hypervolume Contributions of each location
    (0 if the location is not on the Pareto Front)
    """
    d1HypI = np.zeros((d2Points.shape[0], 1))
    # Get the local Pareto Front for each point
    d1Fronts = fastNonDominatedSort(d2Points)
    # Get the HypI for each point
    for i in range(d2Points.shape[0]):
        b1inFront = d1Fronts == d1Fronts[i]+1
        d2ThesePoints = np.vstack((d2Points[b1inFront[:,0], :], d2Points[i,:]))
        # Get the hypervolume of the non-dominated front with all points
        d1HypI[i] = calculateHypervolume(d2ThesePoints, d1Reference)
    return d1HypI

def calculateExpectedHypervolumeContributionMC(d1Mean, d1Std, d2CurrentPoints, d1Reference, iNumSamples=1000):
    """Returns the Expected Hypervolume Contribution for a new location
    with independent Gaussian uncertainty, given an existing exact Pareto Front.
    
    d1Mean          -- a 2-element list specifying the mean location of the new point
    d1Std           -- a 2-element list specifying the standard deviation of uncertainty 
                        on the location of the new point
    d2CurrentPoints -- a 2D array of non-dominated locations (each row is a location)
    d1Reference     -- a 2-element list specifying the nadir point (e.g. [5, 5])
    iNumSamples     -- the number of Monte Carlo samples to draw from the location's 
                        Normal distribution (default = 1000)
    
    Returns a float for the mean Hypervolume Contribution of a new point, given the existing archive
    """
    d2Samples = np.array(np.random.normal(d1Mean[0], d1Std[0], iNumSamples))
    for i in range(1, len(d1Mean)):
        d2Samples = np.vstack([d2Samples,
                               np.random.normal(d1Mean[i], d1Std[i], iNumSamples)])
    d2Samples = d2Samples.T
    # Determine the HVC of each new sample point
    d1Hvc = calculatePotentialHvcsGivenFront(d2Samples, d2CurrentPoints, d1Reference)
    # Return the mean HVC
    return d1Hvc.mean()

def getExcitingNewLocation(d2CurrentArchive, d2InputSpace,
                           d1BoundsMin=None, d1BoundsMax=None, 
                           jitter=0.1):
    """Generates a new location near to an existing non-dominated point.
	
	d2CurrentArchive 	-- a 2D array of locations in solution space (each row is a location)
	d2InputSpace		-- a 2D array of locations in input space (each row is a location)
    d1BoundsMin         -- a 1D array of minimum bounds for each input space dimension
    d1BoundsMax         -- a 1D array of maximum bounds for each input space dimension 
	jitter 				-- the standard deviation as a fraction of the input domain size
	
	Returns a 1D float array for the new location in input space.
	"""
	# Choose a parent location on the Pareto Front
    d2Front, b1OnFront = getNonDominatedFront(d2CurrentArchive)
    d2FrontInput = d2InputSpace[b1OnFront, :]
    iMother = np.random.randint(0, d2Front.shape[0])
    d1Mother = d2FrontInput[iMother, :]
    # Generate a new location a small distance away
    d1NewLocation = []
    for i in range(d2InputSpace.shape[1]):
        #Get range of input space
        dMax = max(d2InputSpace[:, i])
        if d1BoundsMax is not None:
            dMax = d1BoundsMax[i]
        dMin = min(d2InputSpace[:, i])
        if d1BoundsMin is not None:
            dMin = d1BoundsMin[i]
        # Create new random number in range for this dimension
        dRng = np.random.randn() * (dMax - dMin) * jitter
        dNewPos = d1Mother[i] + dRng
        if dNewPos > dMax:
            dNewPos = dMax
        if dNewPos < dMin:
            dNewPos = dMin
        d1NewLocation.append(dNewPos)
    return d1NewLocation

# x_init = np.array([[0.9375, 0.1875],
# [0.1875, 0.5625],[0.5625, 0.8125],
# [0.8125, 0.4375],[0.3125, 0.3125],
# [0.6875, 0.9375],[0.0625, 0.0625],
# [0.4375, 0.6875]])

# y_init = np.array([[0.9375, 1.96633392], [0.1875, 5.06808301],
# [0.5625,6.66982675], [0.8125,2.62364217], 
# [0.3125,2.840573  ], [0.6875,6.62719466], 
# [0.0625,1.19225753], [0.4375, 5.01002008]])

# test = getExcitingNewLocation(y_init, x_init, [0,0], [1,1])
