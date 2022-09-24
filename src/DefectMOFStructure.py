#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:39:33 2022

@author: jace
"""

from cmath import isnan
import numpy as np
import time 
import copy 
import random
import os 

from pymatgen.core.lattice import Lattice
from pymatgen.util.coord import lattice_points_in_supercell
from src.customized_PeriodicSite import PeriodicSite, Site
import pymatgen.core.structure as mgStructure
from pymatgen.util.coord import all_distances

from src.helper import CheckConnectivity, TreeSearch, WarrenCowleyParameter, SwapNeighborList, WriteStructure,substitute_funcGroup
from src.cappingAgent import water, water2, dummy, h2ooh, oh
class DefectMOFStructure():
    
    def __init__(self, mof, superCell, defectConc, numofdefect, linker_type, charge_comp = 0):
        
        self.original_mof = mof
        self.superCell = superCell
        self.defectConc = defectConc
        self.numofdefect = numofdefect
        self.linker_type = linker_type
        self.charge_comp = charge_comp

        self.ReverseMonteCarlo_flag = True

    def __mul__(self, structure, scaling_matrix):
        original_atom_num = len(structure)
        LCA_count = np.nanmax([x.LCA for x in structure if x.LCA is not None])+1
        MCA_count = np.nanmax([x.MCA for x in structure if x.MCA is not None])+1
        scale_matrix = np.array(scaling_matrix, np.int)
        if scale_matrix.shape != (3, 3):
            scale_matrix = np.array(scale_matrix * np.eye(3), np.int)
        new_lattice = Lattice(np.dot(scale_matrix, structure._lattice.matrix))

        f_lat = lattice_points_in_supercell(scale_matrix)
        c_lat = new_lattice.get_cartesian_coords(f_lat)
        _new_sites_seperate_molecular = []
        for i,v in enumerate(c_lat):
            for site in structure:
                if site.NbM is np.nan:
                    site.properties['NbM'] = np.nan
                else:
                    site.properties['NbM'] = site.NbM + i*original_atom_num 
                if site.LCA is np.nan:
                    site.properties['LCA'] = np.nan
                else:
                    site.properties['LCA'] = site.LCA + i*LCA_count
                if site.NbL is np.nan:
                    site.properties['NbL'] = np.nan
                else:
                    site.properties['NbL'] = [ xx + i*original_atom_num for xx in site.NbL]
                if site.MCA is np.nan:
                    site.properties['MCA'] = np.nan
                else:
                    site.properties['MCA'] = site.MCA + i*MCA_count

                s = PeriodicSite(
                    site.species,
                    site.coords + v,
                    new_lattice,
                    properties=site.properties,
                    coords_are_cartesian=True,
                    to_unit_cell=False,
                    skip_checks=True,
                )

                _new_sites_seperate_molecular.append(s)
        new_moleculars = mgStructure.Structure.from_sites(_new_sites_seperate_molecular)
        #new_charge = structure._charge * np.linalg.det(scale_matrix) if structure._charge else None 
        return new_moleculars

    def make_supercell(self, molecular, scaling_matrix, to_unit_cell: bool = True):
        """
        Create a supercell.

        Args:
            scaling_matrix: A scaling matrix for transforming the lattice
                vectors. Has to be all integers. Several options are possible:

                a. A full 3x3 scaling matrix defining the linear combination
                   the old lattice vectors. E.g., [[2,1,0],[0,3,0],[0,0,
                   1]] generates a new structure with lattice vectors a' =
                   2a + b, b' = 3b, c' = c where a, b, and c are the lattice
                   vectors of the original structure.
                b. An sequence of three scaling factors. E.g., [2, 1, 1]
                   specifies that the supercell should have dimensions 2a x b x
                   c.
                c. A number, which simply scales all lattice vectors by the
                   same factor.
            to_unit_cell: Whether or not to fall back sites into the unit cell
        """
        s_sep = self.__mul__(molecular , scaling_matrix)
        
        return s_sep
        
    def Sub_with_Capping_agent(self,neighbor_list):

        # try to cap OMS with something
        # Case 1: general case, based on plane allignment 

        # Case 2: with two metal cluster close to each other
        
        # Case 3: To be added when met

        # Case 4: failed return None, and just provide missing linker without capping agent 


        return
        
    def Build_supercell(self):
        
        t0 = time.time()
        
        """
        build MOF supercell
        """
        self.mof_superCell = self.make_supercell(copy.deepcopy(self.original_mof), self.superCell)

        print("Finish super cell build, which took %f Seconds" % (time.time()-t0))
        
    def DefectGen(self):

        num_of_delete_linkers = self.numofdefect
        self.candidate_index_list = []
        for i,x in enumerate(self.linkers):
            if x.formula == self.linker_type:
                self.candidate_index_list.append(i)
        if num_of_delete_linkers > len(self.candidate_index_list):
            num_of_delete_linkers = len(self.candidate_index_list)
        rand_linker = random.sample(self.candidate_index_list,num_of_delete_linkers)
        self.defectConc = len(rand_linker)/len(self.linkers)
        
        self.neighbor_list = []
        for i,x in enumerate(self.linkers):
            _neighbor_linkers_list_ = []
            _neighbor_nodes_list_ = self.coord_indexes_MetalCluster[i]
            for _node_index_ in _neighbor_nodes_list_:
                _neighbor_linkers_list_.extend( self.linker_indexes_MetalCluster[_node_index_] )
            try:
                _neighbor_linkers_list_.remove(i)
            except:
                print("neighbour list error")
                self.ReverseMonteCarlo_flag = False
            _neighbor_linkers_list_ = list(np.unique(_neighbor_linkers_list_))
            if i in rand_linker:
                _ele_ = [i,'defect', _neighbor_linkers_list_]
            else:
                _ele_ = [i,'normal', _neighbor_linkers_list_]
            self.neighbor_list.append(_ele_)
        
        return
                
        
    def ReverseMonteCarlo(self, candidate_index_list = None, SRO = 0, beta = 1.0, N = 10000 , MaxErr = 0.001):
        '''
        neighbor_list: 2-D list. [[1, 'Yb', ['Nd', 'Yb']]...]
        SRO: target alpha (short range order)
        N: steps number
        beta: smoothfactor for exp(beta|alpha-SRO|)
        '''

        neighbor_list = copy.deepcopy(self.neighbor_list)
        
        #old_neighbor_list=copy.deepcopy(neighbor_list)
        if self.ReverseMonteCarlo_flag == False:
            self.Sub_with_Capping_agent(neighbor_list)    
            print("Finish Coordination env analysis, which took %f Seconds" % (time.time()-t0))
            
            return neighbor_list  
            
        t0 = time.time()
        if candidate_index_list == None:
            candidate_index_list = self.candidate_index_list


        if len(candidate_index_list)<=2:
            return neighbor_list
        n_metal = len(neighbor_list)
        alpha_Yb_0 = WarrenCowleyParameter(neighbor_list, 'normal', 'defect')
        metric_Yb_0 = abs(alpha_Yb_0-SRO)
        alpha_Yb_history = []
        n = 0
        if SRO == None:
            SRO = np.inf
            outputAll = True
            
        while n <= N:# and metric_Yb_0 >= MaxErr:
            n += 1
            ID_1, ID_2 = random.sample(candidate_index_list, 2)
            if neighbor_list[ID_1][1] != neighbor_list[ID_2][1]:
                new_neighbor_list = SwapNeighborList(neighbor_list, ID_1, ID_2)
                alpha_Yb = WarrenCowleyParameter(new_neighbor_list, 'normal', 'defect')
                metric_Yb = abs(alpha_Yb-SRO)
                if metric_Yb < metric_Yb_0:
                    neighbor_list = new_neighbor_list
                    alpha_Yb_0 = alpha_Yb
                    metric_Yb_0 = metric_Yb
                    if metric_Yb_0 <= MaxErr:
                        print(n, alpha_Yb)
                        break
                elif metric_Yb > metric_Yb_0 and np.exp(-beta*(metric_Yb-metric_Yb_0)) >= random.uniform(0, 1):
                    neighbor_list = new_neighbor_list
                    alpha_Yb_0 = alpha_Yb
                    metric_Yb_0 = metric_Yb
                else:
                    new_neighbor_list = neighbor_list
                if alpha_Yb not in alpha_Yb_history:
                    alpha_Yb_history.append(alpha_Yb)
                    print(alpha_Yb)
                
        self.Sub_with_Capping_agent(neighbor_list)    
        print("Finish Coordination env analysis, which took %f Seconds" % (time.time()-t0))
            
        return neighbor_list  
        
