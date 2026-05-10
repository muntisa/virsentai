[![](https://img.shields.io/badge/View%20Project-Website-blue)](https://muntisa.github.io/virsentai)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17445222.svg)](https://doi.org/10.5281/zenodo.17445222)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17450943.svg)](https://doi.org/10.5281/zenodo.17450943)

# Viral Sentry AI (VirSentAI)

**> Automated Zoonotic Surveillance & Drug Repurposing Agent**

<img src="docs/img/virus.png" alt="Virus illustration" width="25%" style="border-radius: 15px;">


Advanced AI protection against human threats by autonomous scanning for new zoonotic viruses and drug repurposing. Publicly funded academic research, developed in the European Union (EU).

VirSentAI is a multimodal agent for zoonotic defense & therapeutic data fusion, an autonomous, tri-model agent designed to close the gap between viral emergence and therapeutic response. It orchestrates a unified surveillance workflow by continuously synthesizing intelligence from three specialized AI architectures:
- Model 1: Gemma4 E4B LLM for context extraction ([gemma-4-E4B-it-Q4_K_M](https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF)),
- Model 2: HyenaDNA for genomic risk prediction ([hyenadna-medium-160k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf)),
- Model 3: [PLAPT](https://github.com/Bindwell/PLAPT) AI, a Transformer using molecular embeddings for chemical affinity screening.

This end-to-end data fusion transforms raw genomic sequences and unstructured text into actionable therapeutic candidates, providing a scalable solution for proactive pandemic preparedness.

Viral Sentry AI is a research prototype with the core as the new VirSentAI model, a unique AI model specifically fine-tuned from the pretrained HyenaDNA architecture for the task of predicting human host tropism. The platform is engineered to automatically and continuously scan public DNA databases for new viral sequences and metadata. For each virus with at least 90% of probability of human infection, the PLAPT pre-trained model is calculated the affinity energy between the viral proteins and the current ChEMBL approved drugs. All the predictions should be verified with other computational methods and experimental methods.

Our mission is to leverage artificial intelligence and bioinformatics to provide actionable insights for researchers, health professionals, and policymakers. In alignment with this goal, the VirSentAI tool is free access and open-source software, with the code publicly available on GitHub.

## File Structure of scripts

### Download raw data

#### NCBI_full_genome_viruses.md
- `raw-data/NCBI/fetch_virus_metadata.py` - Fetch virus metadata from NCBI
- `raw-data/NCBI/batch_download_viruses.py` - Batch download virus genomes
- `raw-data/NCBI/convert_to_tsv.py` - Convert NCBI data to TSV format
- `raw-data/NCBI/download_sample.py` - Download sample virus data
- `read_sqlite.py` - Read/query SQLite database

#### VirHostDB.md
- `raw-data/VirusHostDB/download_data.py` - Download VirusHostDB data
- `raw-data/VirusHostDB/process_data.py` - Process VirusHostDB data

#### BVBRC.md
- `raw-data/BVBRC/ds-3_1-Viprbrc_get_viruses_metadata.py` - API metadata retrieval
- `raw-data/BVBRC/ds-3_2-Viprbrc_filter_viruses_metadata.py` - Metadata filtering and cleaning
- `raw-data/BVBRC/ds-3_3-Viprbrc_get_viruses_data.py` - Genomic sequence download

### TSV to RAW SQLite

#### SQLite_load_raw.md
- `SQL_load_raw_ncbi_from_TSV.py` - Load NCBI/RefSeq data to SQLite
- `SQL-load_virushostdb_from_TSV.py` - Load VirusHostDB data to SQLite
- `SQL-load_raw_bvbrc_from_TSV.py` - Load BVBRC data to SQLite

### Clean database

#### DB-Clean_raw_data.md
- `DB-clean_raw_sqlite.py` - Main cleaning script
- `DB-RemoveFakeViruses.py` - Remove fake/invalid virus records
- `DB-LLMFetchHostUNK.py` - Fetch hosts via LLM for Unknown records (gemma-4-E4B-it-Q4_K_M.gguf)
- `DB-AddCorrectedUNKHosts.py` - Update database with corrected unknown hosts
- `DB-Dashboard.py` - Generate HTML analytics dashboard

### Datasets

#### Dataset.md
- `DS-Filter.py` - Extract and filter virus data from SQLite database
- `DS-Undersampling.py` - Undersampling for balanced datasets
- `DS-Split.py` - Data preparation with MinHash + LSH
- `DS-CheckSimilarityLeakage.py` - Similarity leakage check
- `DS-CheckSplitClasses.py` - Check class distribution in splits

### Training

#### FT-Training_virsentai_v3.md
- `FT-virsentai_v3_1.py` - Fine-tune original HyenaDNA (bin to safetensors)
- `FT-virsentai_v3_2.py` - Improve safetensor model
- `FT-virsentai_v3_3.py` - Further model improvements

### Predictions for Unknown host viruses

#### PRED-UNK_host_viruses.md
- `CreateEmplySQLitePred.py` - Create empty prediction SQLite
- `02_predict_NEW.py` - Predict zoonotic potential for UNK hosts
- `03_add_to_db_NEW.py` - Add UNK predictions to database
- `04_update_webapp.py` - Update webapp with UNK predictions

### Scan new viruses

#### PRED-NewViruses.md
- `01a_scan_viruses_raw_RefSeq.py` - Scan RefSeq for new viruses (save raw data)
- `01b_fetch_unknown_hosts_RefSeq.py` - Find hosts for "Unknown" entries using LLM
- `02_predict_NEW_SCAN.py` - Predict zoonotic potential for new scans
- `03_add_to_db_NEW_SCAN.py` - Add new scanning predictions to database
- `04_update_webapp.py` - Update index HTML with new scans

### Drug repurposing

#### REP-DrugRepurposing.md
- `xplapt.py` - Main PLAPT model for drug repurposing
- `REP-GetApprovedDrugs.py` - Retrieve approved drugs from ChEMBL (filter by MW)
- `REP-drug_repurposing.py` - Calculate affinity energy for zoonotic viruses (prob 0.8, AE 0.9)
- `REP-exportFilteredPLAPT.py` - Update repurposing.html

### Virus Taxonomy

#### TAXON-Dashboard.md
- `DS-GetOrganismTaxonomy.py` - Fetch taxonomy from NCBI
- `DS-FindMissingTaxonomy.py` - Find missing taxonomy entries
- `DS-taxonomy_fetcher.py` - Fetch taxonomy per organism
- `generate_taxonomy_stats.py` - Generate taxonomy stats JSON
- `taxonomy_html.py` - Generate taxonomy dashboard HTML

## Our Team
Cristian R. Munteanu
- AI and Bioinformatics expert, lead programmer, and methodology designer
- Affiliation: CITIC, University of A Coruña, Spain
- ORCID: [0000-0002-5628-2268](https://orcid.org/0000-0002-5628-2268)

Jose Vázquez-Naya
- AI, ontologies, and cybersecurity expert
- Affiliation: CITIC, University of A Coruña, Spain
- ORCID: [0000-0002-6194-5329](https://orcid.org/0000-0002-6194-5329)

Eduardo Tejera
- Bioinformatics expert with experience in virus prediction models.
- Affiliation: Universidad de Las Américas, Quito, Ecuador
- ORCID: [0000-0002-1377-0413](https://orcid.org/0000-0002-1377-0413)
