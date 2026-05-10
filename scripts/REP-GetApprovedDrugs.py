#!/usr/bin/env python3
"""
ChEMBL Approved Drugs Fetcher

Queries ChEMBL database for approved drugs, calculates molecular weight (MW),
and filters based on user-specified MW range.

Usage:
    python REP-GetApprovedDrugs.py --min-mw 200 --max-mw 500

Arguments:
    --min-mw    Minimum molecular weight (Da) [default: 200]
    --max-mw    Maximum molecular weight (Da) [default: 500]

Output:
    - approved_drugs_200_500_MW.tsv (or custom range based on args)
    - Columns: molecule_chembl_id, pref_name, canonical_smiles, MW

Requirements:
    - chembl_webresource_client
    - rdkit for MW calculation
    - config.py with paths

Notes:
    - max_phase=4 ensures only approved drugs
    - Filters out drugs without SMILES structure
    - Calculates MW using RDKit
"""

import pandas as pd
import argparse
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors
from config import *

def get_approved_drugs_with_smiles():
    """
    Queries the ChEMBL database to retrieve a list of approved drugs 
    and their canonical SMILES strings.
    """
    print("Connecting to ChEMBL API...")
    
    molecule = new_client.molecule
    
    try:
        # Define the query:
        # 1. 'max_phase' must be 4 (Approved drug)
        # 2. 'molecule_structures__canonical_smiles__isnull' must be the STRING 'False'
        approved_drugs = molecule.filter(
            max_phase=4, 
            molecule_structures__canonical_smiles__isnull='False'
        ).only([
            'molecule_chembl_id', 
            'pref_name', 
            'molecule_structures'
        ])

        print("Querying data... This may take a moment.")
        
        df = pd.DataFrame(list(approved_drugs))
        
        # Extract the SMILES string from the nested 'molecule_structures' dictionary
        df['canonical_smiles'] = df['molecule_structures'].apply(
            lambda x: x.get('canonical_smiles') if isinstance(x, dict) else None
        )
        
        # Select and rename final columns for clarity
        df = df[['molecule_chembl_id', 'pref_name', 'canonical_smiles']]
        
        print(f"Successfully retrieved {len(df)} approved drugs.")
        return df

    except Exception as e:
        print(f"An error occurred during the ChEMBL query: {e}")
        return None
    
def filter_by_mw(df, min_mw, max_mw):
    """
    Calculates the Molecular Weight (MW) for each drug and filters based on MW range.

    Args:
        df (pd.DataFrame): DataFrame containing 'canonical_smiles' column.
        min_mw (int): Minimum acceptable Molecular Weight in Daltons (Da).
        max_mw (int): Maximum acceptable Molecular Weight in Daltons (Da).

    Returns:
        pd.DataFrame: The filtered DataFrame with an added 'MW' column.
    """
    print(f"Calculating Molecular Weight (MW) for filtering...")
    
    # 1. Create RDKit Mol objects from SMILES strings
    df['mol'] = df['canonical_smiles'].apply(lambda smiles: Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None)
    
    # 2. Calculate Molecular Weight (MW)
    df['MW'] = df['mol'].apply(lambda mol: Descriptors.MolWt(mol) if mol is not None else None)
    
    # 3. Drop rows where MW could not be calculated (invalid SMILES) or is NaN
    df.dropna(subset=['MW'], inplace=True)
    
    initial_count = len(df)
    
    # 4. Filter: Keep drugs where MW is between min_mw and max_mw
    filtered_df = df[(df['MW'] >= min_mw) & (df['MW'] <= max_mw)].copy()
    
    final_count = len(filtered_df)
    removed_count = initial_count - final_count

    print(f"Filtering complete.")
    print(f"   - Initial drug count: {initial_count}")
    print(f"   - Removed (MW < {min_mw} or MW > {max_mw}): {removed_count}")
    print(f"   - Final drug count: {final_count}")
    
    # Drop the temporary 'mol' column
    filtered_df.drop(columns=['mol'], inplace=True)
    
    return filtered_df

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get approved drugs from ChEMBL and filter by molecular weight (MW)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # Default: MW between 200 and 500 Da
              python REP-GetApprovedDrugs.py

              # Custom range: MW between 300 and 600 Da
              python REP-GetApprovedDrugs.py --min-mw 300 --max-mw 600

              # Use short flags
              python REP-GetApprovedDrugs.py -min 150 -max 800
        """
    )
    parser.add_argument(
        "--min-mw", "-min",
        type=float,
        default=DEFAULT_MIN_MW,
        help=f"Minimum molecular weight in Da [default: {DEFAULT_MIN_MW}]"
    )
    parser.add_argument(
        "--max-mw", "-max",
        type=float,
        default=DEFAULT_MAX_MW,
        help=f"Maximum molecular weight in Da [default: {DEFAULT_MAX_MW}]"
    )
    
    args = parser.parse_args()
    
    min_mw = args.min_mw
    max_mw = args.max_mw
    
    print(f"\n{'='*60}")
    print(f"REP-GetApprovedDrugs.py - Filter by MW: {min_mw} - {max_mw} Da")
    print(f"{'='*60}\n")
    
    approved_drugs_df = get_approved_drugs_with_smiles()

    if approved_drugs_df is not None:
        # Run the filtering function
        filtered_drugs = filter_by_mw(approved_drugs_df, min_mw=min_mw, max_mw=max_mw)
        
        print(f"\n--- Filtered Drugs ({min_mw} <= MW <= {max_mw} Da) ---")
        print(filtered_drugs[['pref_name', 'MW']].head())
        
        # Save as TSV file with MW range in filename
        output_file = DRUG_OUTPUT_FILE.format(min_mw=int(min_mw), max_mw=int(max_mw))
        filtered_drugs.to_csv(output_file, sep='\t', index=False)
        print(f"\nData saved to {output_file}")