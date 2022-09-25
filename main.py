#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:48:48 2021

@author: jace

This package is designed to generate initial guess of defect MOFS structures

"""

from src.DefectMOFGenerator import DefectMOFStructureBuilder
import os
from pymatgen.io.cif import CifParser,CifWriter
import shutil
from ase.io import read, write

def main(formatting):
    """
    Default file path system:
        - Input/original cifs: 
            'cifs/'
            
    """
    
    if formatting :

        cif_path = 'cifs'
        cifs = os.listdir(cif_path)
        cifs.sort()

        for cif in cifs:
            mof = read(os.path.join(cif_path, cif))
            write(os.path.join(cif_path, cif), mof)
    
    cifFolder = 'cifs/'
    cifList = os.listdir(cifFolder)
    stat_file = ['archive','cifs','output','main.py','README.md','requirements.txt','src']
    
    for cif in cifList:
        cifName = cif.split('_')[0]
        print(os.path.join(cifFolder,cif))
        try:
            os.mkdir('output/'+cifName)
        except:
            pass

        a = DefectMOFStructureBuilder( os.path.join(cifFolder,cif), output_dir= 'output/'+cifName)
        a.LinkerVacancy()

        #except:
        #    cifName = cifName+'_debug'
            
        # if not os.path.isdir('output/'+cifName):
        #     os.mkdir('output/'+cifName)
            
        # fileList = os.listdir('.')
        # for file in fileList:
        #     if file not in stat_file:
        #         shutil.move( file, 'output/'+cifName)
    
    """
    uncomment for unsucceful cif file read
    """
    #cifFile = CifParser(cifFolder + cifFile_Name)
    #structure = cifFile.get_structures()[0]
    #out = CifWriter(structure)
    #out.write_file('src/'+cifFile_Name)
    

main(True)
# if __name__ == '__main__':
#   main(False)