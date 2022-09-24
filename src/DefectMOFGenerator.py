#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:06:27 2022

@author: Zhenzi Yu
"""

import os
import math
import numpy as np
import copy
import random
import time 

from pymatgen.io.cif import CifParser
from mofid.run_mofid import cif2mofid
import pymatgen.analysis.graphs as mgGraph
import pymatgen.core.bonds  as mgBond
import pymatgen.core.structure as mgStructure
from pymatgen.io.vasp.inputs import Poscar

from src.helper import CheckConnectivity, TreeSearch, WriteStructure, CheckNeighbourMetalCluster
from src.DefectMOFStructure import DefectMOFStructure

from ase.io import read, write

class DefectMOFStructureBuilder():
    
    def __init__(self, cifFile_Name, output_dir = '.' , charge_comp = 0, sepMode = 'MetalOxo', cutoff = 12.8 ):
        
        self.cifFile_Name = cifFile_Name
        self.output_dir = output_dir
        self.sepMode = sepMode
        self.cutoff = cutoff
        self.charge_comp = charge_comp

        self.original_nodes = None
        self.original_linkers, self.original_linkers_length = None, None
        self.processed_structure = None

        self.original_Structure = self.ReadStructure(self.cifFile_Name)
        self.SeperateStructure(self.cifFile_Name)
        
        return
    
    def ReadStructure(self,cifFile_Name):
        
        cifFile = CifParser(cifFile_Name)
        # bug fixed, cannot allow primitive reduction!
        structure = cifFile.get_structures(primitive=False)[0]
        
        return structure
    
    def SeperateStructure(self, cifFileName, linkerSepDir = 'linkerSep' ):

##############################################################################        
#       Coding notes: Metal nodes append after linkers!!!!! 
##############################################################################
# 
#         
        # use the packge to seperate the linker/nodes and read in
        t0 = time.time()
        cif2mofid(cifFileName ,output_path = os.path.join(self.output_dir,linkerSepDir))
        self.original_nodes = self.ReadStructure(os.path.join(linkerSepDir, self.sepMode, 'nodes.cif'))
        self.original_linkers = self.ReadStructure(os.path.join(linkerSepDir, self.sepMode, 'linkers.cif'))
        self.original_linkers_length = len(self.original_linkers.sites)
        print("SBU seperation finished, which takes %f Seconds" % (time.time()-t0))
        
        # Subgraph detection : linker_cluster_assignment - LCA
        linker_cluster_assignment = CheckConnectivity(self.original_linkers)
    
        # Neighbour Metal Cluster detection : NbM
        coord_bond_array, metal_cluster_assignment, coord_bond_list = CheckNeighbourMetalCluster(self.original_linkers, self.original_nodes )
        NbM = [x + len(self.original_linkers) for x in coord_bond_array]
        
        # pass Subgraph detection to the mg structure; 
        # NOTE: this could be combined together with the prvious step, but just for debug purposes, I seperated
        self.original_linkers.add_site_property('NbM',NbM)
        self.original_linkers.add_site_property('LCA',linker_cluster_assignment[1])
        self.original_linkers.add_site_property('NbL',[np.nan]*len(self.original_linkers))
        self.original_linkers.add_site_property('MCA',[np.nan]*len(self.original_linkers))
        self.linker_num = linker_cluster_assignment[0]

        # add property to nodes   
        self.original_nodes.add_site_property('NbM',[np.nan]*len(self.original_nodes)) # add np.nan 
        self.original_nodes.add_site_property('LCA',[np.nan]*len(self.original_nodes)) # add np.nan 
        self.original_nodes.add_site_property('NbL',coord_bond_list) # NbL: neighbourhodd linker
        self.original_nodes.add_site_property('MCA',metal_cluster_assignment[1]) # MCA: Metal Cluster assginement 
        self.original_nodes.add_site_property('is_linker',[False]*len(self.original_nodes))

        # create seperate molecules based on graph 
        self.molecules = []
        indexes = []
        # detect linker type 
        for i in range(self.linker_num):
            index = np.array( np.where(linker_cluster_assignment[1]==i)[0], dtype=int)
            indexes.append(index)
            sites = [ self.original_linkers[i] for i in index ]
            molecule = mgStructure.Structure.from_sites(sites)
            self.molecules.append(molecule)
        formulas = [ x.formula for x in self.molecules]
        self.linker_type = np.unique(formulas) 
        
        # construct new MOF structure: add lable distinguishing Metal/linker, add label for linker type, 
        # Step 0: add unique id for sites. In the late process, sites might be removed, so need to defined an id besides index.
        id_linkers = np.arange(len(self.original_linkers))
        self.original_linkers.add_site_property('fixed_id',id_linkers)
        id_nodes = np.arange(len(self.original_nodes))+len(self.original_linkers)
        self.original_nodes.add_site_property('fixed_id',id_nodes)


        # Step 1: label linker type 
        linker_label = np.full(len(self.original_linkers),None) 
        is_linker = np.full(len(self.original_linkers),True)

        for jj,molecule in enumerate(self.molecules):
            linker_type_num = np.where(self.linker_type == molecule.formula)[0]
            linker_label[indexes[jj]] = linker_type_num
        self.original_linkers.add_site_property('linker_label',linker_label)
        self.original_nodes.add_site_property('linker_label',[np.nan]*len(self.original_nodes))
        self.original_linkers.add_site_property('is_linker',is_linker)

        # Step 2 : add metal site into linker site, combined as a whole 
        processed_structure_sites = []
        for site in self.original_linkers:
            processed_structure_sites.append(site)       
        for site in self.original_nodes:
            processed_structure_sites.append(site)
        self.processed_structure = mgStructure.Structure.from_sites(processed_structure_sites)

        # print and summary
        # TODO: add some check function to make sure the right structure is read into the code 

        print("There are %d atoms in Metal Cluster, and %d atoms in linkers, with %d linkers, %d types" \
        % ( len(self.original_nodes), len(self.original_linkers), self.linker_num, len(np.unique(formulas))))            

        return None

    def _DefectDensityControl_(self):
        
        cell = self.original_Structure.lattice.abc
        min_index = cell.index(min(cell))
        max_index = cell.index(max(cell))
        _mid_index = [0,1,2]
        _mid_index.remove(min_index)
        _mid_index.remove(max_index)
        mid_index = _mid_index[0]
        
        def assign_supercell(cellnum):
            cell = [1,1,1]
            if cellnum == 1:
                return cell
            elif cellnum ==2 or cellnum==3 or cellnum==5:
                cell[min_index] = cellnum
            elif cellnum == 6:
                cell[min_index] = 2
                cell[mid_index] = 3
            elif cellnum == 4:
                cell[min_index] = 2
                cell[mid_index] = 2
            else:
                raise "error for super cell assignment"
            return cell

        enumerate_conc = [1/2,1/3,2/3,1/4,3/4,1/5,2/5,3/5,4/5,1/6,5/6,1,2,3,4,5,6]
        corr_defect_num = [1,1,2,1,3,1,2,3,4,1,5,1,2,3,4,5,6]
        corr_cell = [2,3,3,4,4,5,5,5,5,6,6,1,1,1,1,1,1]
        original_conc = 1/self.linker_num
        desired_conc = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        achived_conc = {}
        
        for _conc_goal_ in desired_conc:
            for i,_conc_possible_ in enumerate(enumerate_conc):
                _conc_possible2_ = original_conc*_conc_possible_
                if abs((_conc_possible2_-_conc_goal_)/_conc_goal_) < 0.3:
                    achived_conc[_conc_possible2_] = [_conc_goal_, corr_defect_num[i], corr_cell[i], assign_supercell(corr_cell[i])]
                    break
        
        
        return achived_conc
    
    def StructureGeneration(self, superCell, defectConc, numofdefect, linker_type):
        
        working_mof = copy.deepcopy(self.processed_structure)

        defect_structure = DefectMOFStructure(working_mof, superCell, defectConc, numofdefect, linker_type, self.charge_comp)
        defect_structure.Build_supercell()
        defect_structure.DefectGen()
        defect_structure.ReverseMonteCarlo()
        
        all_components = defect_structure.linkers.copy()
        all_components.extend(defect_structure.node_clusters)

        return
    
    def LinkerVacancy(self):
        
        self.possible_Defect_Density = self._DefectDensityControl_()
        for i_linker,linker_type in enumerate(self.linker_type):
            for key,val in self.possible_Defect_Density.items():
                print("Currently generate linker vacancy defect with %s, at conc. of %.3f" %(linker_type,key))
                self.StructureGeneration(val[3],key,val[1], i_linker)
        
        return
    
    def DebugVisualization():
        
        return
