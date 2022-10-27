#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:50:37 2022

@author: jace
"""
from cmath import isclose
import numpy as np
import os
import copy

import pymatgen.core.bonds  as mgBond
from pymatgen.io.vasp.inputs import Poscar
import pymatgen.core.structure as mgStructure
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.operations import SymmOp
from scipy.sparse.csgraph import connected_components
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.vis.structure_vtk import StructureVis

import src.cappingAgent as cappingAgent 


#################################################################################################
# important feature to be included:                                                             #
# 1. charge dict for compensation: either through auto method or handcoded dict file            #
# 2. coord bond info based on different metal center                                            #
#################################################################################################

#################################################################################################
# important Note:                                                             
# Lots of things in this file are ugly and need fix
#################################################################################################


def CheckNeighbourMetalCluster(linker, metal):
    
### an improved way to check the connectivity of the atoms  ###
    coord_dict = []
    len1,len2 = len(linker),len(metal)
    
    # array for linker: linker atom can only coord to one metal
    coord_bond_array = np.full(len1, np.nan)
    # dict for metal: one metal can have multiple coord atom
    coord_bond_list = plot_data = [[] for _ in metal]

    # generate the coord bond array
    for i in range(len1):
        for j in range(len2):
            _distance_ = linker[i].distance(metal.sites[j])
            abs_distance = abs(linker[i].frac_coords - metal.sites[j].frac_coords)
            # TODO: no coord bond info in pymatgen, thus use a naive cutoff at 2.8 A, needs to be fixed 
           
            if ((linker[i].specie.value =='O') or (linker[i].specie.value =='N')) and _distance_ < 2.8 and all(abs_distance<0.5):
                coord_bond_array[i] = j
                coord_bond_list[j].append(i)

    # group the metal cluster
    cluster_array = np.zeros((len2,len2))
    for i in range(len2):
        for j in range(i+1,len2):
            if np.linalg.norm(metal[i].coords - metal[j].coords,ord=2) < 4.0:
                cluster_array[i,j] = 1
                cluster_array[j,i] = 1
            else:
                cluster_array[i,j] = 0
    cluster_assignment = connected_components(cluster_array)

    return coord_bond_array, cluster_assignment, coord_bond_list


def CheckConnectivity(linker):
    len1 = len(linker)
    bond_array = np.full((len1,len1), 0, dtype=int)
                   
    for i in range(len1):
        for j in range(i+1,len1):
            no_pbc_check = all( abs(linker[i].frac_coords-linker[j].frac_coords) <0.5 )
            # coord_adjust(linker[i],linker[j]) 
            try:
                isbond = mgBond.CovalentBond.is_bonded(linker[i],linker[j])
            except:
                isbond = False
            if isbond:
                bond_array[i,j] = 1
                bond_array[j,i] = 1          
            else:
                bond_array[i,j] = 0
    assignment = connected_components(bond_array)
    
    return assignment

def coord_adjust(site1,site2):

    frac_coords1, frac_coords2 = site1.frac_coords.copy(), site2.frac_coords.copy()
    diffVec = frac_coords1 - frac_coords2

    for i,diff in enumerate(diffVec):
        if diff > 0.5:
            frac_coords2[i] += 1
        elif diff < -0.5:
            frac_coords1[i] += 1
    site1.frac_coords, site2.frac_coords = frac_coords1, frac_coords2

    return site1, site2



def WarrenCowleyParameter(neighbor_list, center_atom, noncenter_atom):
    '''
    neighbor_list: 2-D list. [[1, 'Yb', ['Nd', 'Yb']]...]
    center_atom: reference atom. 'Yb' or 'Nd'
    '''
    
    # n_neighbor = len(neighbor_list[0][-1])
    # find center atom list
    center_atoms =[]
    for neighbor in neighbor_list:
        if neighbor[1] == center_atom:
            center_atoms.append(neighbor)
    
    # neighbor_list_names = ['neighbor_'+str(i+1) for i in range(n_neighbor)]
    probs = []
    for _index_, _type_, _neighbor_list_ in neighbor_list:
        # list_name = [x[-1][i] for x in center_atoms]
        n_center_atom = [ neighbor_list[x][1] for x in _neighbor_list_ ].count(center_atom)
        n_noncenter_atom = [ neighbor_list[x][1] for x in _neighbor_list_ ].count(noncenter_atom)
        if (n_center_atom + n_noncenter_atom) != len(_neighbor_list_):
            raise "neighbor error"
        # if n_center_atom == 0:
        #     prob = 0
        else:
            prob = n_center_atom /( n_noncenter_atom + n_center_atom )
        probs.append(prob)
    prob_BgivenA = np.average(probs)
    n_center_atom = len(center_atoms)
    n_metal = len(neighbor_list)
    prob_A = n_center_atom/n_metal
    prob_B = 1-prob_A
#    alpha = 1-prob_A*prob_BgivenA/prob_B
    alpha = 1-prob_BgivenA/prob_A
    return alpha

def SwapNeighborList(neighbor_list, ID_1, ID_2):
    '''
    neighbor_list: 2-D list. [[1, 'Yb', ['Nd', 'Yb']]...]
    ID_1, ID_2: the two metals to be swapped
    '''
    new_neighbor_list = copy.deepcopy(neighbor_list)
    #swap center metals
    new_neighbor_list[ID_1][1], new_neighbor_list[ID_2][1] = neighbor_list[ID_2][1], neighbor_list[ID_1][1] 
    #update neighbors
    # for nebr in neighbor_list[ID_1][-1]:
    #     new_neighbor_list[nebr][-1][neighbor_list[nebr-1][-2].index(ID_1)] = neighbor_list[ID_2-1][1] 
    # for nebr in neighbor_list[ID_2-1][-2]:
    #     new_neighbor_list[nebr-1][-1][neighbor_list[nebr-1][-2].index(ID_2)] = neighbor_list[ID_1-1][1]
    return new_neighbor_list
    
def WriteStructure(output_dir, structure, name = 'POSCAR', sort = True):
    
    if sort == True:
        structure.sort()
    out_POSCAR = Poscar(structure=structure)
    # out_POSCAR.selective_dynamics=DynamicsM
    out_POSCAR.write_file(os.path.join(output_dir,name))                
    
    return

def NbMAna(mof, deleted_linker, _index_linker_):

    """
    goal of this helper function: 
        input a seperate linker(specified by _index_linker)
        return the metal-coordLinker-Linker1stneighour pair(three components!) 
    
    return varibles:
        metal_dict_sitebased: fixed_id:site
        MCA_dict_notuniqueid: metal_cluster_num: site
        metal_coord_dict_sitebased: fixed_id:[ (fixed_id, coord sites)]
        metal_coord_neighbour_dict_sitebased: fixed_id: [fixed_id, bonded_sites]
        dist_metal_cluster: dist array between metal clusters
        index_fixed_id_dict : dict from fixed_id to index

    """

    # initiate the return varibles
    metal_dict_sitebased = {}
    metal_coord_dict_sitebased = {}
    metal_coord_neighbour_dict_sitebased = {}
    # fetch the the current index (Compared to the fixed id )
    index_fixed_id_dict = {}
    for i, s in enumerate(mof):
        index_fixed_id_dict[s.fixed_id] = i

    # construct metal - coord dict first
    for index in _index_linker_:
        if ~np.isnan(deleted_linker[index].NbM):
            metal_fixed_id = deleted_linker[index].NbM
            if metal_fixed_id not in metal_dict_sitebased.keys():
                metal_dict_sitebased[metal_fixed_id] = mof[index_fixed_id_dict[metal_fixed_id]]
                metal_coord_dict_sitebased[metal_fixed_id] = [] 
                metal_coord_dict_sitebased[metal_fixed_id].append( (deleted_linker[index].fixed_id, deleted_linker[index]))
        else:
            continue
    
    # based on metal - coord dict, construct coord-neighbour dict 
    for item in metal_coord_dict_sitebased.items():
        coord_index, coord_atoms = item[0], item[1]
        for atom in coord_atoms:
            for _i in _index_linker_:
                try: 
                    isbond = mgBond.CovalentBond.is_bonded(atom[1], deleted_linker[_i])
                except:
                    isbond = False
                if isbond and deleted_linker[_i].fixed_id!=atom[0]:
                    if atom[0] not in metal_coord_neighbour_dict_sitebased.keys():
                        metal_coord_neighbour_dict_sitebased[atom[0]] = []
                        metal_coord_neighbour_dict_sitebased[atom[0]].append((_i,atom[1].distance(deleted_linker[_i])))
                    else:
                        metal_coord_neighbour_dict_sitebased[atom[0]].append((_i,atom[1].distance(deleted_linker[_i])))

    # a very naive way to calculate the min dist between two metal cluster:
    # TODO: could combined with code above 
    associated_NbM = [ deleted_linker[s].NbM for s in _index_linker_]
    coord_metal_index = np.array([i for i,s in enumerate(mof) if s.fixed_id in associated_NbM])
    MCA_dict_notuniqueid = {}
    for index in coord_metal_index:
        if mof[index].MCA not in MCA_dict_notuniqueid.keys():
            MCA_dict_notuniqueid[ mof[index].MCA] = []
            MCA_dict_notuniqueid[ mof[index].MCA].append( (mof[index].fixed_id, mof[index]))
        else:
            MCA_dict_notuniqueid[ mof[index].MCA].append( (mof[index].fixed_id, mof[index]))
    num_metal_cluster = len(MCA_dict_notuniqueid)
    dist_metal_cluster = np.ones((num_metal_cluster,num_metal_cluster))*100
    for ii,key1 in enumerate(MCA_dict_notuniqueid.keys()):
        for jj,key2 in enumerate(MCA_dict_notuniqueid.keys()):
            x, y = MCA_dict_notuniqueid[key1], MCA_dict_notuniqueid[key2]
            for _x in x:
                for _y in y:
                    _dist_ = _x[1].distance(_y[1])
                    if _dist_ < dist_metal_cluster [ii,jj] and ii!=jj:
                        dist_metal_cluster [ii,jj] = _dist_
    
    return metal_dict_sitebased, MCA_dict_notuniqueid, metal_coord_dict_sitebased, metal_coord_neighbour_dict_sitebased, dist_metal_cluster, index_fixed_id_dict


def addOH(metal_site, bonded_site, bonded_site_1stngb):
    M_O = metal_site.coord - bonded_site.coord
    N_O = bonded_site.coord - bonded_site_1stngb.coord

    # rescale the bond length : currently based on Zn-O


def addH2O(metal_dict_sitebased, metal_coord_dict_sitebased,metal_coord_neighbour_dict_sitebased):

    M_O = metal_coord_dict_sitebased[1].coords - metal_dict_sitebased.coords
    N_O= []
    for site in metal_coord_neighbour_dict_sitebased:
        N_O.append( site.coords - metal_coord_dict_sitebased[1].coords)

    O_M_bond_length = 2.1365
    O_coords = metal_dict_sitebased.coords + O_M_bond_length/np.linalg.norm(M_O,2)*M_O
    O_site = copy.deepcopy(metal_coord_dict_sitebased[1])
    O_site.species, O_site.coords = 'O', O_coords

    O_H_bond_length =  0.97856
    H_coords1 = metal_coord_dict_sitebased[1].coords + O_H_bond_length/np.linalg.norm(N_O[0],2)*N_O[0]
    H_sites1 = copy.deepcopy(metal_coord_neighbour_dict_sitebased[0])
    H_sites1.species, H_sites1.coords = 'S', H_coords1

    H_coords2 = metal_coord_dict_sitebased[1].coords + O_H_bond_length/np.linalg.norm(N_O[1],2)*N_O[1]
    H_sites2 = copy.deepcopy(metal_coord_neighbour_dict_sitebased[1])
    H_sites2.species, H_sites2.coords = 'S', H_coords2

    return [O_site, H_sites1, H_sites2]
    # rescale the bond length : currently based on Zn-O


def addHOHOH(metals, coord_atom):

    if len(coord_atom[0])!=1 or len(coord_atom[1])!=1:
        print("ERROR: for HOHOH, not unique coord O, nothing added")
        return []

    site_O1 = coord_atom[0][0]
    site_O2 = coord_atom[1][0]

    O_M1 = coord_atom[0][0][1].coords - metals[0].coords
    O_M2 = coord_atom[1][0][1].coords - metals[1].coords
    M_mid = (metals[0].coords + metals[1].coords)/2

    length_M_H_mid = 2.638178855934525
    H_mid = copy.deepcopy(site_O1)[1]
    H_mid.species, H_mid.coords = 'S', M_mid + length_M_H_mid/np.linalg.norm(O_M1 + O_M2,2)*(O_M1 + O_M2)

    OH_length = 0.97
    H_left = copy.deepcopy(site_O1)[1]
    vector = H_mid.coords - site_O1[1].coords
    H_left.species, H_left.coords = 'S', site_O2[1].coords  + 0.97/np.linalg.norm(vector,2)*vector

    H_right = copy.deepcopy(site_O1)[1]
    vector = H_mid.coords - site_O2[1].coords
    H_right.species, H_right.coords = 'S', site_O1[1].coords  + 0.97/np.linalg.norm(vector,2)*vector



    return [site_O1[1], site_O2[1], H_mid, H_left, H_right]
    # rescale the bond length : currently based on Zn-O


def DebugVisualization(vis_structure):
        vis = StructureVis()
        vis.set_structure(vis_structure)
        vis.show()


def nodes_expansion(structure):
    
    newsites = []

    for site in structure.sites:
        permutationList = [[],[],[]]
        for i,pos in enumerate(site.frac_coords):
            if pos > 0.5:
                permutationList[i].extend([pos])
                permutationList[i].extend([pos-1])
            else:
                permutationList[i].extend([pos])
                permutationList[i].extend([pos+1])
        for x in permutationList[0]:
            for y in permutationList[1]:
                for z in permutationList[2]:
                    frac_coords = [x,y,z]
                    new_site = copy.deepcopy(structure[0])
                    new_site.frac_coords = frac_coords
                    newsites.append(new_site)

    new_structure = mgStructure.Structure.from_sites(newsites)

    return new_structure
