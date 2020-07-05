# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:37:47 2020

@author: Clym Stock-Williams
"""
import unittest
import numpy as np
import ParetoFrontND as pf

class BasicParetoFrontFunctionTests(unittest.TestCase):
    """ Runs tests of N-D functions for domination and Pareto Front identification
    """
    d1Reference4D = np.array([2.0, 2.0, 2.0, 2.0])

    d2Archive4D = np.array([
        [1.0, 1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0]])

    def test_isDominatedBy_4D_strictlyEqual_isFalse(self):
        '''Equal locations do not dominate each other'''
        d1PointTest = self.d1Reference4D * 1.0
        # Run and assert
        result = pf.isDominatedBy(self.d1Reference4D, d1PointTest)
        self.assertFalse(result)

    def test_isDominatedBy_4D_oneBetter_restWorse_isFalse(self):
        '''Domination implies all locations are at least equal and one is better'''
        d1PointTest = self.d1Reference4D * 1.0
        for i in range(len(d1PointTest)):
            d1PointTest[i] += 0.01
        d1PointTest[1] -= 0.02
        # Run and assert
        result = pf.isDominatedBy(self.d1Reference4D, d1PointTest)
        self.assertFalse(result)

    def test_isDominatedBy_4D_oneBetter_restEqual_isTrue(self):
        '''Domination implies all locations are at least equal and one is better'''
        d1PointTest = self.d1Reference4D * 1.0
        d1PointTest[1] -= 0.01
        # Run and assert
        result = pf.isDominatedBy(self.d1Reference4D, d1PointTest)
        self.assertTrue(result)

    def test_getNonDominatedFront_4D_newPointDominated_returnsOriginalFront(self):
        '''The Pareto Front is the set of mutually non-dominated points'''
        d1NewPoint = self.d2Archive4D[0, :] * 1.0
        for i in range(len(d1NewPoint)):
            d1NewPoint[i] += 0.01
        d2Archive = np.vstack([self.d2Archive4D, d1NewPoint])
        # Run and assert
        result, _ = pf.getNonDominatedFront(d2Archive)
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(all(result[0] == self.d2Archive4D[0, :]))
        self.assertTrue(all(result[1] == self.d2Archive4D[1, :]))

    def test_getNonDominatedFront_4D_newPointNotDominated_returnsLargerFront(self):
        '''The Pareto Front is the set of mutually non-dominated points'''
        d1NewPoint = self.d2Archive4D[0, :] * 1.0
        d1NewPoint[0] -= 0.01
        d1NewPoint[2] += 0.01
        d2Archive = np.vstack([self.d2Archive4D, d1NewPoint])
        # Run and assert
        result, _ = pf.getNonDominatedFront(d2Archive)
        self.assertEqual(result.shape[0], 3)
        self.assertTrue(all(result[0] == self.d2Archive4D[0, :]))
        self.assertTrue(all(result[1] == self.d2Archive4D[1, :]))
        self.assertTrue(all(result[2] == d1NewPoint))

    def test_getNonDominatedFront_4D_newPointDominatesOne_returnsNewFront(self):
        '''The Pareto Front is the set of mutually non-dominated points'''
        d1NewPoint = self.d2Archive4D[0, :] * 1.0
        d1NewPoint[0] -= 0.01
        d2Archive = np.vstack([self.d2Archive4D, d1NewPoint])
        # Run and assert
        result, _ = pf.getNonDominatedFront(d2Archive)
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(all(result[0] == self.d2Archive4D[1, :]))
        self.assertTrue(all(result[1] == d1NewPoint))

    def test_getNonDominatedFront_4D_newPointDominatesBoth_returnsNewFront(self):
        '''The Pareto Front is the set of mutually non-dominated points'''
        d1NewPoint = self.d2Archive4D[0, :] * 1.0
        for i in range(len(d1NewPoint)):
            if self.d2Archive4D[1, i] < d1NewPoint[i]:
                d1NewPoint[i] = self.d2Archive4D[1, i] - 0.01
            else:
                d1NewPoint[i] -= 0.01
        d2Archive = np.vstack([self.d2Archive4D, d1NewPoint])
        # Run and assert
        result, _ = pf.getNonDominatedFront(d2Archive)
        self.assertEqual(result.shape[0], 1)
        self.assertTrue(all(result[0] == d1NewPoint))
        
    def test_fastNonDominatedSort_4D_twoFronts_returnsCorrectly(self):
        d1NewPoint = self.d2Archive4D[0, :] * 1.0
        d1NewPoint[0] -= 0.01
        d2Archive = np.vstack([self.d2Archive4D, d1NewPoint])
        d1NewPoint = self.d2Archive4D[0, :] * 1.0
        d1NewPoint[1] += 0.01
        d2Archive = np.vstack([d2Archive, d1NewPoint])
        # Run and assert
        result = pf.fastNonDominatedSort(d2Archive)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 1)
        self.assertEqual(result[2], 1)
        self.assertEqual(result[3], 3)

class IgdTests(unittest.TestCase):
    d2TruePF2D = np.array([
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0]])

    d2Archive2D = np.array([
        [0.2, 1.0],
        [0.9, 0.8]])
    
    def test_calculateIgd_returnsCorrectValue(self):
        result = pf.calculateIGD(self.d2TruePF2D, self.d2Archive2D)
        self.assertAlmostEqual(np.sqrt(0.94)/3., result)
    
class HypervolumeFunctionTests(unittest.TestCase):
    """ Runs tests of 2-D functions for calculating hypervolume """
    d1Reference2D = np.array([2.0, 2.0])

    d2Archive2D = np.array([
        [1.0, -1.0],
        [-1.0, 1.0]])

    def test_calculateHypervolume_returnsCorrectValue(self):
        '''Area between Pareto Front and reference (nadir) point.'''
        result = pf.calculateHypervolume(self.d2Archive2D, self.d1Reference2D)
        self.assertEqual(5.0, result)

    def test_calculateHypervolumeContributions_newDominatedPoint_returnsCorrectly(self):
        '''Dominated points cannot contribute to or affect hypervolume'''
        d1ExtraPoint = self.d1Reference2D * 1.0
        d1ExtraPoint[0] -= 0.01
        d1ExtraPoint[1] -= 0.01
        d2Archive = np.vstack([self.d2Archive2D, d1ExtraPoint])
        result = pf.calculateHypervolumeContributions(d2Archive, self.d1Reference2D)
        self.assertEqual(2.0, result[0])
        self.assertEqual(2.0, result[1])
        self.assertEqual(0.0, result[2])

    def test_calculateHypervolumeContributions_newNonDominatedPoint_returnsCorrectly(self):
        '''Area between updated Pareto Front and reference (nadir) point.'''
        d1ExtraPoint = np.array([0.0, 0.0])
        d2Archive = np.vstack([self.d2Archive2D, d1ExtraPoint])
        result = pf.calculateHypervolumeContributions(d2Archive, self.d1Reference2D)
        self.assertEqual(1.0, result[0])
        self.assertEqual(1.0, result[1])
        self.assertEqual(1.0, result[2])

    def test_calculateHypervolumeContributions_newNonDominatedBadPoint_returnsCorrectly(self):
        '''Points which are worse than the reference (nadir) in any dimension
        cannot contribute to or affect hypervolume.'''
        d1ExtraPoint = self.d1Reference2D * -1.0
        d1ExtraPoint[0] = self.d1Reference2D[0] + 0.5
        d2Archive = np.vstack([self.d2Archive2D, d1ExtraPoint])
        result = pf.calculateHypervolumeContributions(d2Archive, self.d1Reference2D)
        self.assertEqual(2.0, result[0])
        self.assertEqual(2.0, result[1])
        self.assertEqual(0.0, result[2])

    def test_calculateNegativeHypervolumeContributions_newDominatedPoint_returnsCorrectly(self):
        '''Area between Pareto Front and new point as reference.'''
        d1ExtraPoint = self.d1Reference2D * 1.0
        d2Archive = np.vstack([self.d2Archive2D, d1ExtraPoint])
        result = pf.calculateNegativeHypervolumeContributions(d2Archive)
        self.assertEqual(0.0, result[0])
        self.assertEqual(0.0, result[1])
        self.assertEqual(5.0, result[2])

    def test_calculateNegativeHypervolumeContributions_newNonDominatedPoint_returnsCorrectly(self):
        '''Area between Pareto Front and new point as reference.'''
        d1ExtraPoint = np.array([0.0, 0.0])
        d2Archive = np.vstack([self.d2Archive2D, d1ExtraPoint])
        result = pf.calculateNegativeHypervolumeContributions(d2Archive)
        self.assertEqual(0.0, result[0])
        self.assertEqual(0.0, result[1])
        self.assertEqual(0.0, result[2])

    def test_calculatePotentialHvcsGivenFront_returnsCorrectly(self):
        '''Calculate hypervolume contribution for each point separately.'''
        d2ExtraPoints = np.vstack([
            [-1.5, 2.5],    # Bad
            [-0.5, 1.5],    # Dominated
            [-0.5, 0.5],    # New PF
            [0.0, 0.0],     # New PF
            [0.5, -0.5],    # New PF
            [1.5, -0.5],    # Dominated
            [2.5, -1.5]])   # Bad
        result = pf.calculatePotentialHvcsGivenFront(d2ExtraPoints,
                                                     self.d2Archive2D,
                                                     self.d1Reference2D)
        self.assertEqual(0.0, result[0])
        self.assertEqual(0.0, result[1])
        self.assertEqual(0.75, result[2])
        self.assertEqual(1.0, result[3])
        self.assertEqual(0.75, result[4])
        self.assertEqual(0.0, result[5])
        self.assertEqual(0.0, result[6])

    def test_calculatePotentialNegativeHvcsGivenFront_returnsCorrectly(self):
        '''Calculate negative hypervolume contribution for each point separately.'''
        d2ExtraPoints = np.vstack([
            [-1.5, 2.5],    # Bad
            [-0.5, 1.5],    # Dominated
            [-0.5, 0.5],    # New PF
            [1.5, 1.5],     # Dominated
            [1.5, -0.5],    # Dominated
            [2.5, -1.5]])   # Bad
        result = pf.calculatePotentialNegativeHvcsGivenFront(d2ExtraPoints,
                                                             self.d2Archive2D)
        self.assertEqual(0.0, result[0])
        self.assertEqual(0.25, result[1])
        self.assertEqual(0.0, result[2])
        self.assertEqual(2.25, result[3])
        self.assertEqual(0.25, result[4])
        self.assertEqual(0.0, result[5])
        
    def test_calculateHypIs_returnsCorrectly(self):
        d2ExtraPoints = np.vstack([
            [-0.5, 1.5],    # Dominated - F2
            [-0.5, 0.5],    # New PF
            [0.5, 0.5],     # Dominated - F2
            [0.5, -0.5],    # New PF
            [1.0, 1.0],     # Dominated - F3
            [1.5, 0.0]])    # Dominated - F2
        d2Archive = np.vstack([self.d2Archive2D, d2ExtraPoints])
        result = pf.calculateHypIs(d2Archive, self.d1Reference2D)
        self.assertEqual(4.25, result[0])
        self.assertEqual(4.0, result[1])
        self.assertEqual(1.75, result[2])
        self.assertEqual(4.0, result[3])
        self.assertEqual(2.25, result[4])
        self.assertEqual(4.25, result[5])
        self.assertEqual(1.0, result[6])
        self.assertEqual(1.5, result[7])

    def test_calculateExpectedHypervolumeContributionMC_newNonDominatedPoint_smallVar_returnsCorrectly(self):
        '''Calculate the mean hypervolume contribution with a perfect PF and
        Gaussian-distributed new location.'''
        point_mean = np.array([0.0, 0.0])
        point_std = np.array([1e-9, 1e-9])
        result = pf.calculateExpectedHypervolumeContributionMC(point_mean, point_std,
                                                                self.d2Archive2D, self.d1Reference2D)
        self.assertAlmostEqual(1.0, result)

    def test_calculateExpectedHypervolumeContributionMC_newNonDominatedPoint_largeVar_returnsCorrectly(self):
        '''Calculate the mean hypervolume contribution with a perfect PF and
        Gaussian-distributed new location.'''
        point_mean = np.array([0.0, 0.0])
        point_std = np.array([1.0, 1.0])
        result = pf.calculateExpectedHypervolumeContributionMC(point_mean, point_std,
                                                                self.d2Archive2D, self.d1Reference2D)
        self.assertLess(1.0, result)

    def test_calculateExpectedHypervolumeContributionMC_newDominatedPoint_smallVar_returnsCorrectly(self):
        '''Calculate the mean hypervolume contribution with a perfect PF and
        Gaussian-distributed new location.'''
        point_mean = np.array([1.5, 1.5])
        point_std = np.array([1e-9, 1e-9])
        result = pf.calculateExpectedHypervolumeContributionMC(point_mean, point_std,
                                                                self.d2Archive2D, self.d1Reference2D)
        self.assertAlmostEqual(0.0, result)

    def test_calculateExpectedHypervolumeContributionMC_newDominatedPoint_largeVar_returnsCorrectly(self):
        '''Calculate the mean hypervolume contribution with a perfect PF and
        Gaussian-distributed new location.'''
        point_mean = np.array([1.5, 1.5])
        point_std = np.array([1.0, 1.0])
        result = pf.calculateExpectedHypervolumeContributionMC(point_mean, point_std,
                                                                self.d2Archive2D, self.d1Reference2D)
        self.assertLess(0.0, result)

class ParetoFrontExplorationTests(unittest.TestCase):
    """ Runs tests of functions to assist with acquisition function optimisation """
	x_init = np.array([[0.9375, 0.1875],
	[0.1875, 0.5625],[0.5625, 0.8125],
	[0.8125, 0.4375],[0.3125, 0.3125],
	[0.6875, 0.9375],[0.0625, 0.0625],
	[0.4375, 0.6875]])
	
	y_init = np.array([[0.9375, 1.96633392], [0.1875, 5.06808301],
	[0.5625,6.66982675], [0.8125,2.62364217], 
	[0.3125,2.840573  ], [0.6875,6.62719466], 
	[0.0625,1.19225753], [0.4375, 5.01002008]])
	
	def test_getExcitingLocation_isClose(self):
		'''Returns a point close to current non-dominated front.'''
		test = pf.getExcitingNewLocation(y_init, x_init, [0,0], [1,1])


unittest.main()
