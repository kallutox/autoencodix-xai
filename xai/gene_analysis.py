from helper_functions import *
from gprofiler import GProfiler
import re
import mygene
import ast
from io import StringIO

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_gene_lists_cf(beta):
    if beta == 0.01:
        beta = '001'
    # dls genes
    with open(f"cf_reports/all_features_{beta}_deepliftshap.txt", "r") as file:
        lines = [line.strip() for line in file if line.strip()]
    if lines and not lines[0].startswith('{'):
        lines = lines[1:]

    dls_genes = []
    for line in lines:
        try:
            metadata_str, score_str = line.rsplit(",", 1)
            metadata = ast.literal_eval(metadata_str)
            gene_name = metadata.get('feature_name')
            dls_genes.append(gene_name)
        except Exception as e:
            print("Error processing line:", line)
            print(e)

    # lime genes
    with open(f"cf_reports/all_features_{beta}_lime.txt", "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    if lines and not lines[0].startswith('{'):
        lines = lines[1:]

    lime_genes = []
    for line in lines:
        try:
            metadata_str, score_str = line.rsplit(",", 1)
            metadata = ast.literal_eval(metadata_str)
            gene_name = metadata.get('feature_name')
            lime_genes.append(gene_name)
        except Exception as e:
            print("Error processing line:", line)
            print(e)

    # ig genes
    with open(f"cf_reports/all_features_{beta}_integrated_gradients.txt", "r") as file:
        lines = [line.strip() for line in file if line.strip()]
    if lines and not lines[0].startswith('{'):
        lines = lines[1:]

    ig_genes = []
    for line in lines:
        try:
            metadata_str, score_str = line.rsplit(",", 1)
            metadata = ast.literal_eval(metadata_str)
            gene_name = metadata.get('feature_name')
            ig_genes.append(gene_name)
        except Exception as e:
            print("Error processing line:", line)
            print(e)

    return dls_genes, lime_genes, ig_genes


def get_gene_lists_tcga(beta, cancer_type):
    if beta == 0.01:
        beta = '001'

    # dls genes
    with open(f"tcga_reports/all_features_{cancer_type}_{beta}_deepliftshap.txt", "r") as file:
        dls_list = [line.strip() for line in file if line.strip()]

    dls_genes_with_mod = [line.split(',')[0] for line in dls_list[1:]]
    dls_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in dls_list[1:]
    ]

    # lime genes
    with open(f"tcga_reports/all_features_{cancer_type}_{beta}_lime.txt", "r") as file:
        lime_list = [line.strip() for line in file if line.strip()]

    lime_genes_with_mod = [line.split(',')[0] for line in lime_list[1:]]
    lime_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in lime_list[1:]
    ]

    # ig genes
    with open(f"tcga_reports/all_features_{cancer_type}_{beta}_integrated_gradients.txt", "r") as file:
        ig_list = [line.strip() for line in file if line.strip()]

    ig_genes_with_mod = [line.split(',')[0] for line in ig_list[1:]]
    ig_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in ig_list[1:]
    ]

    return dls_genes, lime_genes, ig_genes


def get_all_gene_ranks_cf(beta, gene_list):
    dls_genes, lime_genes, ig_genes = get_gene_lists_cf(beta)

    print('\nDeepLiftShap Results: ')
    for gene in gene_list:
        get_gene_rank(gene, dls_genes)
    calculate_average_gene_rank(gene_list, dls_genes)

    print('\nLIME Results: ')
    for gene in gene_list:
        get_gene_rank(gene, lime_genes)
    calculate_average_gene_rank(gene_list, lime_genes)

    print('\nIntegradient Gradient Results: ')
    for gene in gene_list:
        get_gene_rank(gene, ig_genes)
    calculate_average_gene_rank(gene_list, ig_genes)

    #add average positions



def calculate_average_gene_rank(candidate_genes, ranked_genes):
    """
    For each gene in candidate_genes, find its 1-indexed rank in ranked_genes.
    Print the rank for each gene (or indicate if the gene is not found), and
    compute the average rank of all genes that are present in ranked_genes.

    Parameters:
    - candidate_genes (list): List of gene names to look for.
    - ranked_genes (list): Ranked list of gene names.

    Returns:
    - float or None: The average rank of found genes, or None if no genes were found.
    """
    positions = []
    for gene in candidate_genes:
        if gene in ranked_genes:
            position = ranked_genes.index(gene) + 1  # convert to 1-indexed rank
            print(f"{gene} position: {position}")
            positions.append(position)
        else:
            print(f"{gene} is not in the list.")

    if positions:
        average_rank = sum(positions) / len(positions)
        print(f"Average gene rank: {average_rank:.2f}")
        return average_rank
    else:
        print("No genes found in the ranked list.")
        return None


def get_all_gene_ranks_tcga(cancer_type, beta, gene_list):
    dls_genes, lime_genes, ig_genes = get_gene_lists_tcga(beta, cancer_type)


    print('\nDeepLiftShap Results: ')
    for gene in gene_list:
        get_gene_rank(gene, dls_genes)
    calculate_average_gene_rank(gene_list, dls_genes)

    print('\nLIME Results: ')
    for gene in gene_list:
        get_gene_rank(gene, lime_genes)
    calculate_average_gene_rank(gene_list, lime_genes)

    print('\nIntegradient Gradient Results: ')
    for gene in gene_list:
        get_gene_rank(gene, ig_genes)
    calculate_average_gene_rank(gene_list, ig_genes)


def get_gene_info_cf(beta, gene_n):
    dls_genes, lime_genes, ig_genes = get_gene_lists_cf(beta)
    dls_genes = dls_genes[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]


    mg = mygene.MyGeneInfo()
    print("DeepLiftShap - Genes:")
    dls_gene_info = mg.querymany(dls_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in dls_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")

    print("LIME - Genes:")
    lime_gene_info = mg.querymany(lime_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in lime_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")

    print("Integrated Gradients - Genes:")
    ig_gene_info = mg.querymany(ig_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in ig_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")


def get_gene_info_tcga(beta, cancer_type, gene_n):
    dls_genes, lime_genes, ig_genes = get_gene_lists_tcga(beta, cancer_type)
    dls_genes = dls_genes[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]

    # Retrieve gene information using mygene
    mg = mygene.MyGeneInfo()
    print("DeepLiftShap - Genes:")
    dls_gene_info = mg.querymany(dls_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in dls_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")

    print("LIME - Genes:")
    lime_gene_info = mg.querymany(lime_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in lime_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")

    print("Integrated Gradients - Genes:")
    ig_gene_info = mg.querymany(ig_genes, scopes='symbol', fields='name,summary,go', species='human')
    for gene in ig_gene_info:
        symbol = gene['query']
        name = gene.get('name', 'No name available')
        summary = gene.get('summary', 'No summary available')
        print(f"{symbol} - {name}\nSummary: {summary}\n")


def pathway_enrichment_analysis_cf(beta, gene_n):
    dls_genes, lime_genes, ig_genes = get_gene_lists_cf(beta)
    dls_genes = dls_genes[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]

    if beta == 0.01:
        beta = '001'

    background = get_interim_data(f"cf_{beta}_1")
    feature_names = list(background.columns)
    ensembl_ids = [gene_id.replace("RNA_", "") for gene_id in feature_names]
    gene_metadata = get_cf_metadata(ensembl_ids)
    all_genes = [gene_metadata[id]['feature_name'] for id in ensembl_ids if id in gene_metadata]

    gp = GProfiler(return_dataframe=True)

    print("DeepLiftShap - Genes:")
    dls_enrichment_results = gp.profile(organism='hsapiens', query=dls_genes, background=all_genes)
    print(dls_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])
    #sorted_results = dls_enrichment_results.sort_values(by='intersection_size', ascending=False)
    #print(sorted_results[['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    print("LIME - Genes:")
    lime_enrichment_results = gp.profile(organism='hsapiens', query=lime_genes, background=all_genes)
    print(lime_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    print("Integrated Gradients - Genes:")
    ig_enrichment_results = gp.profile(organism='hsapiens', query=ig_genes, background=all_genes)
    print(ig_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    dls_enrichment_results.to_csv(f'evaluation_reports/cf_dls_enrichment_{beta}_top{gene_n}.csv', index=False)
    lime_enrichment_results.to_csv(f'evaluation_reports/cf_lime_enrichment_{beta}_top{gene_n}.csv', index=False)
    ig_enrichment_results.to_csv(f'evaluation_reports/cf_ig_enrichment_{beta}_top{gene_n}.csv', index=False)


def pathway_enrichment_analysis_tcga(beta, cancer_type, gene_n):
    dls_genes, lime_genes, ig_genes = get_gene_lists_tcga(beta, cancer_type)
    dls_genes = dls_genes[:gene_n]
    lime_genes = lime_genes[:gene_n]
    ig_genes = ig_genes[:gene_n]

    if beta == 0.01:
        beta = '001'

    background = get_interim_data(f"tcga_{beta}_1")
    entrez_ids = list(background.columns)
    gene_metadata = get_tcga_metadata(entrez_ids)
    background = [gene_metadata.get(gene_id, gene_id) for gene_id in background.columns]
    all_genes = [
        re.sub(r"\s*\(.*\)$", "", line.split(',')[0])
        for line in background[1:]
    ]

    gp = GProfiler(return_dataframe=True)

    print("DeepLiftShap - Genes:")
    dls_enrichment_results = gp.profile(organism='hsapiens', query=dls_genes, background=all_genes)
    print(dls_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])
    #print(dls_enrichment_results.columns.to_list())

    print("LIME - Genes:")
    lime_enrichment_results = gp.profile(organism='hsapiens', query=lime_genes, background=all_genes)
    print(lime_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    print("Integrated Gradients - Genes:")
    ig_enrichment_results = gp.profile(organism='hsapiens', query=ig_genes, background=all_genes)
    print(ig_enrichment_results[
              ['source', 'name', 'description', 'p_value', 'intersection_size', 'parents']])

    dls_enrichment_results.to_csv(f'evaluation_reports/tcga_dls_enrichment_{cancer_type}_{beta}_top{gene_n}.csv', index=False)
    lime_enrichment_results.to_csv(f'evaluation_reports/tcga_lime_enrichment_{cancer_type}_{beta}_top{gene_n}.csv', index=False)
    ig_enrichment_results.to_csv(f'evaluation_reports/tcga_ig_enrichment_{cancer_type}_{beta}_top{gene_n}.csv', index=False)


if __name__ == "__main__":
    # cystic fibrosis
    # gene list from malacards
    cf_associated_genes_15 = ['CFTR', 'TGFB1', 'FCGR2A', 'CLN6', 'SERPINA1', 'HFE', 'SCNN1B', 'SCNN1A', 'ELN', 'CAV1',
                              'DNAH5', 'DNAH11', 'CCDC40', 'DNAI1', 'CCDC39']

    # tcga
    # gene lists from malacard
    laml_associated_genes_15 = ['CEBPA', 'DNMT3A', 'JAK2', 'GATA2', 'TERT', 'FLT3', 'RUNX1', 'NPM1', 'KIT', 'KRAS',
                                'ETV6', 'LPP', 'CHIC2', 'MLLT10', 'NUP214']
    prad_associated_genes_15 = ['TP53', 'SPOP', 'PTEN', 'AKT1', 'PIK3CA', 'ERBB2', 'CTNNB1', 'BRAF', 'CDKN2A', 'IDH1',
                                'SMAD4', 'HRAS', 'MAP2K1', 'MED12', 'XPO1']
    thca_associated_genes_15 = ['RET', 'HRAS', 'KRAS', 'LRRC56', 'BRAF', 'NRAS', 'PTEN', 'APC', 'TP53', 'PRKAR1A',
                                'KIT', 'DICER1', 'WRN', 'PTCSC3', 'MALAT1']

    cancer_types = ['LAML', 'PRAD', 'THCA']

    # get ranks of disease associated genes
    #get_all_gene_ranks_cf(beta=0.01, gene_list=cf_associated_genes_15)
    #get_all_gene_ranks_tcga('THCA', 0.01, thca_associated_genes_15)

    # functional gene analysis
    #get_gene_info_cf(beta=0.01, gene_n=15)
    #get_gene_info_tcga(beta=0.01, cancer_type='LAML', gene_n=15)

    # pathway enrichment analysis
    #pathway_enrichment_analysis_cf(1, 100)
    pathway_enrichment_analysis_tcga(0.01, 'THCA', 100)

