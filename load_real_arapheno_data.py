import os
import json
import ssl
import urllib.request
import numpy as np
import torch

def fetch_real_arapheno_dataset(num_target_phenotypes=20, num_target_samples=100):
    """
    Fetches real phenotype measurements directly from the public AraPheno REST API
    (https://arapheno.1001genomes.org/rest/) and constructs a real Arabidopsis phenotype matrix.
    """
    print("Fetching phenotype catalog from AraPheno REST API...")
    ssl_context = ssl._create_unverified_context()
    catalog_url = "https://arapheno.1001genomes.org/rest/phenotype/list.json"
    
    req = urllib.request.Request(catalog_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
        catalog = json.loads(response.read().decode())

    print(f"Total available AraPheno phenotypes cataloged: {len(catalog)}")

    # Filter phenotypes that have sufficient measurements (e.g., >= 200 values)
    suitable_phenotypes = [p for p in catalog if p.get("num_values", 0) >= 200]
    print(f"Phenotypes with >= 200 accession measurements: {len(suitable_phenotypes)}")

    selected_phenotypes = suitable_phenotypes[:num_target_phenotypes]
    
    accession_data = {} # accession_id -> {phenotype_idx: value}
    phenotype_names = []

    for p_idx, p_info in enumerate(selected_phenotypes):
        p_id = p_info["phenotype_id"]
        p_name = p_info.get("name", f"Phenotype_{p_id}")
        phenotype_names.append(p_name)
        safe_p_name = p_name.encode('ascii', 'replace').decode('ascii')
        print(f"[{p_idx+1}/{num_target_phenotypes}] Fetching AraPheno trait '{safe_p_name}' (ID: {p_id})...", flush=True)
        
        url = f"https://arapheno.1001genomes.org/rest/phenotype/{p_id}/values.json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_context, timeout=8) as resp:
                vals = json.loads(resp.read().decode())
                for item in vals:
                    acc_id = item.get("accession_id")
                    val = item.get("phenotype_value")
                    if acc_id is not None and val is not None and not np.isnan(val):
                        if acc_id not in accession_data:
                            accession_data[acc_id] = {}
                        accession_data[acc_id][p_idx] = float(val)
        except Exception as e:
            print(f"  Warning: Could not fetch phenotype {p_id}: {e}", flush=True)

    accession_coverage = [(acc, len(p_dict)) for acc, p_dict in accession_data.items()]
    accession_coverage.sort(key=lambda x: x[1], reverse=True)
    
    top_accessions = [acc for acc, count in accession_coverage[:num_target_samples]]
    print(f"Selected top {len(top_accessions)} real Arabidopsis accessions.", flush=True)

    # Build Y matrix (num_samples x num_target_phenotypes)
    Y_mat = np.zeros((len(top_accessions), len(selected_phenotypes)), dtype=np.float32)
    
    for i, acc_id in enumerate(top_accessions):
        p_dict = accession_data[acc_id]
        for j in range(len(selected_phenotypes)):
            if j in p_dict:
                Y_mat[i, j] = p_dict[j]
            else:
                col_vals = [p_dict_inner[j] for acc, p_dict_inner in accession_data.items() if j in p_dict_inner]
                Y_mat[i, j] = np.mean(col_vals) if col_vals else 0.0

    # Save real matrix locally
    os.makedirs("data", exist_ok=True)
    np.save("data/real_arapheno_Y.npy", Y_mat)
    with open("data/real_phenotype_names.json", "w", encoding="utf-8") as f:
        json.dump(phenotype_names, f, indent=2, ensure_ascii=False)

    print(f"Successfully downloaded and stored AraPheno Phenotype Matrix shape: {Y_mat.shape}", flush=True)
    return top_accessions, phenotype_names, Y_mat

if __name__ == '__main__':
    fetch_real_arapheno_dataset(num_target_phenotypes=15, num_target_samples=100)
